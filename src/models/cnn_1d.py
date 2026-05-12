"""
1D Convolutional Neural Network for light curve classification.

This model operates directly on the 1D time-series (processed light
curve) without phase folding. It must find tiny transit dips buried
in 2,001 noisy flux measurements.

Architecture (tuned for small datasets ~300 samples):
    Input (1, 2001) → Conv1D blocks → Global Avg Pool → FC → Sigmoid

Key fixes over the original version:
    - Smaller architecture (16→32→64 filters instead of 32→64→128)
      to reduce overfitting with limited training data
    - Wider kernels (15,11,7) to capture the full transit shape
      (a typical transit spans ~50-200 time points)
    - Class-weighted BCE loss so the model can't cheat by predicting
      all-positive or all-negative
    - Cosine annealing learning rate scheduler for smoother convergence
    - Gradient clipping to prevent exploding gradients
    - Per-epoch diagnostic logging (gradient norms, prediction distribution)
      so you can see what backpropagation is actually doing

"""

import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from src.utils.config import CNN_1D, MODELS_DIR
from src.features.transforms import augment_light_curve


class LightCurveDataset(Dataset):
    """PyTorch dataset for 1D light curves."""

    def __init__(self, flux_arrays: list, labels: np.ndarray,
                 augment: bool = False):
        self.flux_arrays = flux_arrays
        self.labels = labels.astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        flux = self.flux_arrays[idx].astype(np.float32)
        if self.augment:
            flux = augment_light_curve(flux)
            flux = flux.astype(np.float32)
        # Shape: (1, length) — 1 channel for Conv1d
        x = torch.tensor(flux, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


class CNN1DNet(nn.Module):
    """
    1D CNN architecture (tuned for small datasets).

    Each block: Conv1D → BatchNorm → ReLU → MaxPool
    Then: Global Average Pooling → Dropout → FC → Sigmoid

    Wider kernels (15, 11, 7) capture the full transit shape.
    Fewer filters (16, 32, 64) reduce parameter count to ~14K
    instead of ~52K, reducing overfitting on 300 training samples.
    """

    def __init__(self, input_length: int = 2001,
                 n_filters: list = None, kernel_sizes: list = None,
                 pool_size: int = 2, dropout: float = 0.3,
                 fc_units: int = 128):
        super().__init__()

        layers = []
        in_channels = 1  # Single-channel input (flux only)

        for n_filt, k_size in zip(n_filters, kernel_sizes):
            layers.extend([
                nn.Conv1d(in_channels, n_filt, kernel_size=k_size,
                          padding=k_size // 2),
                nn.BatchNorm1d(n_filt),
                nn.ReLU(),
                nn.MaxPool1d(pool_size),
            ])
            in_channels = n_filt

        self.conv_blocks = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_filters[-1], fc_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_units, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.classifier(x)
        return x.squeeze(-1)


class CNN1DModel:
    """
    Training wrapper with class weighting, LR scheduling, gradient
    clipping, and diagnostic logging.

    Follows the same interface as NaiveBayes and Logistic:
    fit(), predict(), predict_proba(), save(), load().
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = CNN1DNet(
            n_filters=CNN_1D["n_filters"],
            kernel_sizes=CNN_1D["kernel_sizes"],
            pool_size=CNN_1D["pool_size"],
            dropout=CNN_1D["dropout"],
            fc_units=CNN_1D["fc_units"],
        ).to(self.device)
        self.name = "CNN_1D"
        self.config = CNN_1D
        self.threshold = 0.5  # default, overridden by tune_threshold()

    def fit(self, train_flux: list, train_labels: np.ndarray,
            val_flux: list = None, val_labels: np.ndarray = None):
        """
        Train with class weighting, LR scheduling, and early stopping.

        What's different from a naive training loop:

        1. CLASS-WEIGHTED LOSS:
           BCELoss normally treats each sample equally. If 60% of samples
           are planets, the easiest way to minimize loss is to predict
           "planet" for everything (which is what the old model did).
           pos_weight upweights the loss on false positives so the model
           is penalized equally for both types of mistakes.

        2. COSINE ANNEALING LR SCHEDULER:
           Instead of a fixed learning rate, follow a cosine curve
           from the initial LR down to near-zero. This lets the model
           take big steps early (explore) and small steps late (refine).

        3. GRADIENT CLIPPING:
           Caps the gradient norm at 1.0 to prevent exploding gradients.
           Without this, a single bad batch can send weights to infinity.

        4. DIAGNOSTIC LOGGING:
           Every 5 epochs, print gradient norms and prediction stats
           so you can see if backpropagation is actually working.
        """
        train_ds = LightCurveDataset(train_flux, train_labels, augment=True)
        train_dl = DataLoader(train_ds, batch_size=self.config["batch_size"],
                              shuffle=True)

        val_dl = None
        if val_flux is not None:
            val_ds = LightCurveDataset(val_flux, val_labels, augment=False)
            val_dl = DataLoader(val_ds, batch_size=self.config["batch_size"])

        # --- Class-weighted loss ---
        # Count class balance and compute weight
        n_pos = train_labels.sum()
        n_neg = len(train_labels) - n_pos
        if n_pos > 0 and n_neg > 0:
            # pos_weight > 1 means "penalize missed negatives more"
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(self.device)
            print(f"    Class balance: {int(n_pos)} planets, {int(n_neg)} FP → pos_weight={pos_weight.item():.2f}")
        else:
            pos_weight = torch.tensor([1.0], dtype=torch.float32).to(self.device)

        # BCEWithLogitsLoss combines sigmoid + BCE and supports pos_weight.
        # This means remove the Sigmoid from the network's last layer
        # during training and let the loss handle it. But since net
        # already has Sigmoid, use a weighted BCE manually.
        criterion = nn.BCELoss(reduction='none')

        optimizer = torch.optim.Adam(self.net.parameters(),
                                     lr=self.config["learning_rate"],
                                     weight_decay=self.config.get("weight_decay", 1e-4))

        # --- Cosine annealing scheduler ---
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config["epochs"],
            eta_min=self.config.get("eta_min", 1e-6)
        )

        val_metric = self.config.get("val_metric", "loss")
        best_val_score = float("-inf")  # higher is better (use -loss or pr_auc)
        patience_counter = 0
        best_state = None

        for epoch in range(self.config["epochs"]):
            # --- Training ---
            self.net.train()
            train_loss = 0.0
            all_train_preds = []

            for X_batch, y_batch in train_dl:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                preds = self.net(X_batch)

                # Apply class weighting manually
                raw_loss = criterion(preds, y_batch)
                weights = torch.where(y_batch == 1, 1.0, pos_weight.item())
                loss = (raw_loss * weights).mean()

                loss.backward()

                # --- Gradient clipping ---
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),
                                               max_norm=self.config.get("grad_clip_norm", 1.0))

                optimizer.step()
                train_loss += loss.item()
                all_train_preds.extend(preds.detach().cpu().numpy().tolist())

            train_loss /= len(train_dl)
            scheduler.step()

            # --- Validation ---
            if val_dl is not None:
                val_loss, val_pr_auc = self._evaluate_val(
                    val_dl, criterion, pos_weight, val_labels)
                val_score = val_pr_auc if val_metric == "pr_auc" else -val_loss

                if val_score > best_val_score:
                    best_val_score = val_score
                    patience_counter = 0
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.net.state_dict().items()}
                else:
                    patience_counter += 1

                # --- Diagnostic logging every 5 epochs ---
                if (epoch + 1) % 5 == 0:
                    # Gradient norms — are gradients flowing?
                    grad_norms = []
                    for p in self.net.parameters():
                        if p.grad is not None:
                            grad_norms.append(p.grad.norm().item())
                    avg_grad = np.mean(grad_norms) if grad_norms else 0

                    # Prediction distribution — is the model actually discriminating?
                    pred_arr = np.array(all_train_preds)
                    pred_mean = pred_arr.mean()
                    pred_std = pred_arr.std()
                    pct_above_05 = (pred_arr > 0.5).mean() * 100

                    current_lr = scheduler.get_last_lr()[0]

                    print(f"    Epoch {epoch+1:3d}: "
                          f"train_loss={train_loss:.4f}  "
                          f"val_loss={val_loss:.4f}  "
                          f"val_pr_auc={val_pr_auc:.4f}  "
                          f"lr={current_lr:.2e}  "
                          f"grad={avg_grad:.4f}  "
                          f"preds={pred_mean:.3f}±{pred_std:.3f}  "
                          f"{pct_above_05:.0f}%>0.5  "
                          f"patience={patience_counter}/{self.config['patience']}  "
                          f"[stopping on {val_metric}]")

                if patience_counter >= self.config["patience"]:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

        # Restore best model
        if best_state is not None:
            self.net.load_state_dict(best_state)

        # --- Auto-tune threshold on validation set ---
        if val_flux is not None and self.config.get("threshold") == "auto":
            self.tune_threshold(val_flux, val_labels)
        elif isinstance(self.config.get("threshold"), (int, float)):
            self.threshold = float(self.config["threshold"])

        return self

    def tune_threshold(self, val_flux: list, val_labels: np.ndarray):
        """
        Find the classification threshold that maximizes F1 on the
        validation set by scanning the precision-recall curve.

        Why not just use 0.5?
        The model's probability outputs aren't calibrated — especially
        with class weighting, outputs can cluster well below 0.5 while
        still ranking planets above false positives. The optimal threshold
        depends on the actual output distribution.

        """
        from sklearn.metrics import precision_recall_curve, f1_score

        probs = self.predict_proba(val_flux)

        precisions, recalls, thresholds = precision_recall_curve(val_labels, probs)

        # Compute F1 at each threshold
        f1_scores = []
        for p, r in zip(precisions[:-1], recalls[:-1]):
            if p + r > 0:
                f1_scores.append(2 * p * r / (p + r))
            else:
                f1_scores.append(0)

        if f1_scores:
            best_idx = np.argmax(f1_scores)
            self.threshold = float(thresholds[best_idx])
            best_f1 = f1_scores[best_idx]
            print(f"    Threshold tuned: {self.threshold:.4f} "
                  f"(F1={best_f1:.4f} on val set, "
                  f"vs F1={f1_score(val_labels, (probs >= 0.5).astype(int), zero_division=0):.4f} at 0.5)")
        else:
            self.threshold = 0.5
            print(f"    Threshold tuning failed, using default 0.5")

    def predict(self, flux_arrays: list) -> np.ndarray:
        probs = self.predict_proba(flux_arrays)
        return (probs >= self.threshold).astype(int)

    def predict_proba(self, flux_arrays: list) -> np.ndarray:
        self.net.eval()
        ds = LightCurveDataset(flux_arrays, np.zeros(len(flux_arrays)))
        dl = DataLoader(ds, batch_size=self.config["batch_size"])
        all_probs = []
        with torch.no_grad():
            for X_batch, _ in dl:
                X_batch = X_batch.to(self.device)
                probs = self.net(X_batch).cpu().numpy()
                all_probs.append(probs)
        return np.concatenate(all_probs)

    def _evaluate_val(self, dataloader, criterion, pos_weight, val_labels):
        """Evaluate val set — returns (loss, pr_auc) in one forward pass."""
        from sklearn.metrics import average_precision_score
        self.net.eval()
        total_loss = 0.0
        all_probs = []
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                preds = self.net(X_batch)
                raw_loss = criterion(preds, y_batch)
                weights = torch.where(y_batch == 1, 1.0, pos_weight.item())
                total_loss += (raw_loss * weights).mean().item()
                all_probs.extend(preds.cpu().numpy().tolist())
        val_loss = total_loss / len(dataloader)
        val_pr_auc = average_precision_score(val_labels, np.array(all_probs))
        return val_loss, val_pr_auc

    def save(self, path: Path = None):
        if path is None:
            path = MODELS_DIR / "cnn_1d.pt"
        torch.save(self.net.state_dict(), path)

    def load(self, path: Path = None):
        if path is None:
            path = MODELS_DIR / "cnn_1d.pt"
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        return self
