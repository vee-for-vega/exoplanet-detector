.PHONY: install test download preprocess train evaluate all clean

# ============================================================
# SETUP
# ============================================================

install:
	pip install -r requirements.txt

# ============================================================
# TESTING — Run this first!
# ============================================================

test:
	pytest tests/ -v --tb=short

# ============================================================
# DATA PIPELINE
# ============================================================

# Step 1: Download TCE metadata table only (~30 seconds)
download:
	python -m src.data.download

# Step 2a: Download small subset of light curves (~10 min)
download-lc:
	python -m src.data.download --lightcurves --limit 300

# Step 2b: Download ALL light curves (hours — optional stretch goal)
download-lc-all:
	python -m src.data.download --lightcurves

preprocess:
	python -m src.data.preprocess

phase-fold:
	python -m src.data.phase_fold

# ============================================================
# FULL PIPELINE
# ============================================================

all: install test download preprocess phase-fold
	@echo "Pipeline complete. Ready for training."

clean:
	rm -rf data/raw/* data/processed/* data/images/*
	rm -rf models/* results/*
	@echo "Cleaned all generated data."
