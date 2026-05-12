"""
2D Convolutional Neural Network for phase-folded image classification.

This is the primary model — it classifies the 2D images created by
phase_fold.py. This is a standard image classification architecture,
which means everything you know about CNNs from computer vision
applies directly.

Architecture:
    Input (1, 64, 64) → Conv2D blocks → Global Avg Pool → FC → Sigmoid

Why 2D CNN on images vs. 1D CNN on raw curves:
- 2D CNN can detect 2D spatial patterns (the U-shaped transit dip
  has both width in phase and depth in flux)
- Standard CV architectures and transfer learning become available
- Visualizing learned filters is more intuitive (what shapes
  does the model look for?)

Trade-off: The phase-folding step introduces information loss
(binning) and requires a known period. The 1D CNN doesn't need
a known period. Compare both and discuss.
"""

import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from src.utils.config import CNN_2D, PHASE_FOLD_IMAGE_SIZE, MODELS_DIR
from src.features.transforms import augment_image


class PhaseImageDataset(Dataset):
    """PyTorch dataset for phase-folded images."""

    def __init__(self, images: list, labels: np.ndarray,
                 augment: bool = False):
        self.images = images
        self.labels = labels.astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32)
        if self.augment:
            img = augment_image(img)
            img = img.astype(np.float32)
        # Shape: (1, H, W) — single channel grayscale
        x = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


class CNN2DNet(nn.Module):
    """
    2D CNN for phase-folded image classification.

    Same pattern as 1D: Conv blocks → Global Pool → FC → Sigmoid.
    Uses 3x3 kernels (standard in CV) since transit features
    in the phase-folded image are relatively small.
    """

    def __init__(self, image_size: tuple = PHASE_FOLD_IMAGE_SIZE,
                 n_filters: list = None, kernel_sizes: list = None,
                 pool_size: int = 2, dropout: float = 0.3,
                 fc_units: int = 128):
        super().__init__()

        layers = []
        in_channels = 1

        for n_filt, k_size in zip(n_filters, kernel_sizes):
            layers.extend([
                nn.Conv2d(in_channels, n_filt, kernel_size=k_size,
                          padding=k_size // 2),
                nn.BatchNorm2d(n_filt),
                nn.ReLU(),
                nn.MaxPool2d(pool_size),
            ])
            in_channels = n_filt

        self.conv_blocks = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
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
        x = self.global_pool(x).squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        return x.squeeze(-1)


class CNN2DModel:
    """Training wrapper — same interface as all other models."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = CNN2DNet(
            n_filters=CNN_2D["n_filters"],
            kernel_sizes=CNN_2D["kernel_sizes"],
            pool_size=CNN_2D["pool_size"],
            dropout=CNN_2D["dropout"],
            fc_units=CNN_2D["fc_units"],
        ).to(self.device)
        self.name = "CNN_2D"
        self.config = CNN_2D
        self.threshold = 0.5  # default, overridden by tune_threshold()

    def fit(self, train_images: list, train_labels: np.ndarray,
            val_images: list = None, val_labels: np.ndarray = None):
        """Train with class weighting, LR scheduling, and early stopping."""
        train_ds = PhaseImageDataset(train_images, train_labels, augment=True)
        train_dl = DataLoader(train_ds, batch_size=self.config["batch_size"],
                              shuffle=True)

        val_dl = None
        if val_images is not None:
            val_ds = PhaseImageDataset(val_images, val_labels, augment=False)
            val_dl = DataLoader(val_ds, batch_size=self.config["batch_size"])

        # --- Class-weighted loss ---
        n_pos = train_labels.sum()
        n_neg = len(train_labels) - n_pos
        if n_pos > 0 and n_neg > 0:
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(self.device)
            print(f"    Class balance: {int(n_pos)} planets, {int(n_neg)} FP → pos_weight={pos_weight.item():.2f}")
        else:
            pos_weight = torch.tensor([1.0], dtype=torch.float32).to(self.device)

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
            self.net.train()
            train_loss = 0.0
            for X_batch, y_batch in train_dl:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                preds = self.net(X_batch)

                # Weighted loss
                raw_loss = criterion(preds, y_batch)
                weights = torch.where(y_batch == 1, 1.0, pos_weight.item())
                loss = (raw_loss * weights).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),
                                               max_norm=self.config.get("grad_clip_norm", 1.0))
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_dl)
            scheduler.step()

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

                if (epoch + 1) % 5 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"    Epoch {epoch+1:3d}: train_loss={train_loss:.4f}  "
                          f"val_loss={val_loss:.4f}  val_pr_auc={val_pr_auc:.4f}  "
                          f"lr={current_lr:.2e}  "
                          f"patience={patience_counter}/{self.config['patience']}  "
                          f"[stopping on {val_metric}]")

                if patience_counter >= self.config["patience"]:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

        # --- Auto-tune threshold on validation set ---
        if val_images is not None and self.config.get("threshold") == "auto":
            self.tune_threshold(val_images, val_labels)
        elif isinstance(self.config.get("threshold"), (int, float)):
            self.threshold = float(self.config["threshold"])

        return self

    def tune_threshold(self, val_images: list, val_labels: np.ndarray):
        """Find threshold that maximizes F1 on validation set."""
        from sklearn.metrics import precision_recall_curve, f1_score

        probs = self.predict_proba(val_images)
        precisions, recalls, thresholds = precision_recall_curve(val_labels, probs)

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

    def predict(self, images: list) -> np.ndarray:
        probs = self.predict_proba(images)
        return (probs >= self.threshold).astype(int)

    def predict_proba(self, images: list) -> np.ndarray:
        self.net.eval()
        ds = PhaseImageDataset(images, np.zeros(len(images)))
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
            path = MODELS_DIR / "cnn_2d.pt"
        torch.save(self.net.state_dict(), path)

    def load(self, path: Path = None):
        if path is None:
            path = MODELS_DIR / "cnn_2d.pt"
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        return self
    