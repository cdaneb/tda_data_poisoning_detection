# Topological Data Analysis for Data Poisoning Detection in Online Machine Learning Systems

A research-oriented Python framework for detecting data poisoning
attacks in **online machine learning systems** using **Topological Data
Analysis (TDA)** and persistent homology.

This project implements a streaming experiment pipeline combining:

-   Online learning via `SGDClassifier`
-   Sliding-window persistent homology (Ripser)
-   PCA-based dimensionality reduction
-   Robust baseline calibration (median/MAD)
-   Empirical quantile thresholding
-   Clean baseline validation on CICIDS2017

The framework is designed for reproducible, publication-quality
evaluation.

------------------------------------------------------------------------

# Overview

Modern online learning systems are vulnerable to gradual data poisoning
attacks.\
This repository provides:

-   A streaming ML pipeline
-   A TDA-based anomaly monitor
-   Clean baseline calibration tools
-   Synthetic and CICIDS2017 stream support
-   Reproducible experiment artifacts

------------------------------------------------------------------------

# Core Features

## Online Learning

-   `SGDClassifier(loss="log_loss")`
-   Incremental `partial_fit`
-   Time-respecting stream order
-   StandardScaler fit on initial training prefix

## TDA Monitoring

-   Sliding-window point clouds
-   Vietoris--Rips persistent homology via `ripser`
-   H0 and H1 summaries:
    -   `max_persistence`
    -   `persistence_count`
    -   `persistence_entropy`
-   PCA dimensionality reduction before PH
-   Robust z-score calibration (median/MAD)
-   Empirical quantile thresholding
-   Per-window detection with consecutive-flag logic

## Clean Baseline Pipeline

-   Warmup windows for PCA + robust calibration
-   Calibration windows for empirical threshold estimation
-   Frozen threshold for detection phase
-   Window-level metrics and artifacts

## Poisoning Simulation (Synthetic)

-   Label-flip attack
-   Trigger/backdoor-style attack
-   Configurable poisoning windows and rates

------------------------------------------------------------------------

# Installation

From project root:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# Running Tests

Always run tests under the project virtual environment:

``` bash
python -m pytest -q
```

All tests should pass before running experiments.

------------------------------------------------------------------------

# Project Structure

    tda_code/
    ├── README.md
    ├── requirements.txt
    ├── scripts/
    │   └── cleanup_repo.py
    ├── src/
    │   ├── data.py
    │   ├── homology.py
    │   ├── summaries.py
    │   ├── viz.py
    │   ├── sliding_window.py
    │   ├── streaming/
    │   │   ├── online_model.py
    │   │   ├── window_buffer.py
    │   │   ├── tda_monitor.py
    │   │   ├── baseline.py
    │   │   ├── cicids2017.py
    │   │   ├── stream.py
    │   │   ├── drift.py
    │   │   └── poison.py
    │   └── experiments/
    │       ├── run_experiment.py
    │       └── run_clean_baseline.py
    ├── notebooks/
    ├── tests/
    └── outputs/

------------------------------------------------------------------------

# Running Experiments

## Synthetic Streaming Experiment

``` bash
python -m src.experiments.run_experiment
```

Override parameters:

``` bash
python -m src.experiments.run_experiment \
  --n_steps 3000 \
  --poison_rate 0.5 \
  --poison_mode label_flip \
  --point_cloud_mode residuals \
  --output_dir outputs/experiment_run_1
```

------------------------------------------------------------------------

# Clean Baseline (CICIDS2017)

To generate a publication-quality clean baseline:

``` bash
python -m src.experiments.run_clean_baseline --seed 0 --day Monday --max-rows 20000
```

Outputs are written to:

    outputs/clean/seed_<seed>/

Example:

    outputs/clean/seed_0/

------------------------------------------------------------------------

# Clean Baseline Artifacts

## window_metrics.csv

One row per evaluation window including:

-   `t`, `window_id`
-   Model metrics: `accuracy`, `balanced_accuracy`, `precision_1`,
    `recall_1`, `f1_1`, `tp`, `fp`, `tn`, `fn`, `attack_prevalence`
-   TDA features: `h0_max_persistence`, `h0_count`, `h0_entropy`,
    `h1_max_persistence`, `h1_count`, `h1_entropy`
-   Detection fields: `anomaly_score`, `threshold_used`,
    `flagged_window`, `consecutive_flags`, `h1_nonempty`
-   PCA metadata: `pca_requested`, `pca_effective`, `pca_clamped`,
    `pca_disabled`, `dr_method`

------------------------------------------------------------------------

## baseline_params.json

Self-contained snapshot of calibration parameters and run summary.

------------------------------------------------------------------------

# Interpretation of Clean Baseline

A stable clean baseline should exhibit:

-   Low false positive rate (\<1--5%)
-   Stable anomaly score distribution
-   Consistent H1 presence frequency
-   No explosive heavy-tail behavior
-   Balanced accuracy near 1.0 for all-benign slices

------------------------------------------------------------------------

# Repository Cleanup

To remove generated artifacts safely:

``` bash
python scripts/cleanup_repo.py
```

Default behavior removes:

-   `__pycache__/`
-   `.pytest_cache/`
-   `.ipynb_checkpoints/`
-   `*.pyc`

Optional destructive flags:

-   `--remove-results`
-   `--remove-outputs`
-   `--remove-vscode`

------------------------------------------------------------------------

# Research Status

## Current state

-   PCA-based dimensionality reduction integrated
-   Robust median/MAD calibration implemented
-   Empirical quantile thresholding implemented
-   Clean baseline validated (low FPR)
-   Full artifact reproducibility enabled
-   Test suite passing

## Next phase

-   Baseline on attack-present CICIDS slices
-   Controlled poisoning experiments
-   Comparative analysis (H0 vs H1 behavior)
-   Sensitivity to window size and PCA dimension
-   Formal statistical evaluation

------------------------------------------------------------------------

# Dependencies

-   scikit-learn
-   ripser
-   persim
-   numpy
-   pandas
-   matplotlib
-   pytest

------------------------------------------------------------------------

# License

Research / academic use.
