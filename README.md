# TDA Pipeline: Topological Data Analysis

A Python pipeline for Topological Data Analysis (TDA) using persistent homology. Built for toy datasets with extension to sliding-window streaming analysis.

## Features

- **Synthetic data**: `make_circles`, `make_blobs`, random points; optional `standardize` via `preprocess_point_cloud`
- **Persistent homology**: Vietoris-Rips via ripser (H0, H1)
- **Visualization**: Persistence diagrams via persim
- **Summaries**: Max persistence, persistence count, persistence entropy, `summarize_by_dimension`
- **Sliding windows**: Takens embedding for 1D time series; windowed point clouds for streaming

## Install

```bash
pip install -r requirements.txt
```

## Run tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Run from the project root (`tda_code/`).

## Usage

### Toy pipeline (make_circles)

```python
from src import generate_circles, compute_persistence, plot_point_cloud, plot_persistence_diagram
from src.summaries import max_persistence, persistence_count, persistence_entropy

X, y = generate_circles(n_samples=200, noise=0.05, factor=0.5, random_state=42)
result = compute_persistence(X, maxdim=1)
dgms = result['dgms']

# Summaries
h1 = dgms[1]
print(max_persistence(h1), persistence_count(h1, 0.1), persistence_entropy(h1))
```

### Sliding window (Takens embedding)

```python
from src.sliding_window import takens_embedding, sliding_window_persistence

signal = np.cos(5 * np.linspace(0, 20, 1000))
embedded, result = sliding_window_persistence(signal, dimension=3, time_delay=8, stride=10)
```

### Streaming point clouds

```python
from src.data import generate_circles, generate_blobs
from src.homology import compute_persistence
from src.summaries import summarize_by_dimension

# Chunks 1-4, 6-10: circles. Chunk 5: blobs (poison).
for i, X in enumerate(chunks):
    result = compute_persistence(X, maxdim=1)
    s = summarize_by_dimension(result['dgms'], threshold=0.1)
```

## Project structure

```
tda_code/
├── README.md
├── requirements.txt
├── src/
│   ├── data.py           # generate_circles, generate_random, generate_blobs, preprocess_point_cloud
│   ├── homology.py       # ripser wrapper
│   ├── viz.py            # point cloud + diagram plots
│   ├── summaries.py      # max persistence, count, entropy, summarize_by_dimension
│   └── sliding_window.py # Takens embedding, sliding windows
├── notebooks/
│   ├── 01_toy_pipeline.ipynb   # make_circles end-to-end
│   ├── 02_sliding_window_demo.ipynb  # sine wave + Takens
│   ├── 03_streaming_ml_demo.ipynb   # Takens + poisoning
│   ├── 04_streaming_pointclouds.ipynb  # windowed point clouds
│   └── 05_visualize_toy_datasets.ipynb # seaborn + circles/random/blobs
└── tests/
    └── test_pipeline.py
```

## Extending to streaming data

- **1D signals**: Use `takens_embedding` on rolling windows; compute persistence per window.
- **Point cloud streams**: Buffer last W frames; run `compute_persistence` on each batch.

### End-to-end streaming poisoning experiment

This repository also includes a simple end-to-end experiment combining:

- **Online learner**: Logistic regression via `SGDClassifier` wrapped by `OnlineLearner`.
- **Data stream**: Synthetic binary classification stream from `make_classification_stream`.
- **Local poisoning attack**: `PoisoningAttack` with label-flip or trigger-based modes.
- **TDA monitor**: `TDAMonitor` over a `WindowBuffer` with either feature-only or residual-based point clouds.

Run the default experiment from the project root:

```bash
python -m src.experiments.run_experiment
```

or override a few key parameters:

```bash
python -m src.experiments.run_experiment \
  --n_steps 3000 \
  --poison_rate 0.5 \
  --poison_mode label_flip \
  --point_cloud_mode residuals \
  --output_dir outputs/experiment_run_1
```

**Poisoning modes**:

- **`label_flip`**: flips labels inside a time window with probability `poison_rate`. Optionally restricted to a `target_class`. This simulates noisy/poisoned labels in the training stream.
- **`trigger`**: adds a small constant `trigger_value` to selected `trigger_dims` in the feature vector and forces the label to `poison_target_label`. This mimics a backdoor/trigger-style attack.

**Point-cloud modes for TDA**:

- **`features`**: the point cloud is just the recent feature vectors \(x\) in the window; topology reflects geometry of the input space.
- **`residuals`**: each point is `[x, y_prob, err]` where `y_prob` is the model's predicted probability for class 1 and `err` is the 0/1 classification error; topology reflects both data and model behaviour.

**Outputs** (written under `output_dir`, default `outputs/`):

- `config.json`: full `ExperimentConfig` used for the run.
- `metrics.csv`: time series with `t`, `test_accuracy`, and recent poisoning rate.
- `tda_features.csv`: per-window TDA summaries (`h0_*`, `h1_*`) with detection `score` and `flag`.
- `detections.csv`: subset of `tda_features.csv` where detection `flag` is `True`.
- `accuracy_over_time.png`: test accuracy vs time with the poisoning window shaded.
- `detection_score_over_time.png`: detection score vs threshold over time, poisoning window shaded.
- `tda_features_over_time.png`: selected TDA features (e.g. `h1_max_persistence`, `h0_max_persistence`) vs time, poisoning window shaded.

### Run clean baseline

For a publication-quality *clean* baseline on CICIDS2017 with robust calibration and empirical thresholding, run from the project root:

```bash
python -m src.experiments.run_clean_baseline --seed 0 --day Monday --max-rows 20000
```

This writes artifacts under `outputs/clean/seed_<seed>/`, for example `outputs/clean/seed_0/`:

- `window_metrics.csv`: one row per evaluation window with:
  - `t`, `window_id`
  - model metrics: `accuracy`, `balanced_accuracy`, `precision_1`, `recall_1`, `f1_1`, `tp`, `fp`, `tn`, `fn`, `attack_prevalence`
  - TDA features: `h0_max_persistence`, `h0_count`, `h0_entropy`, `h1_max_persistence`, `h1_count`, `h1_entropy`
  - detection fields: `anomaly_score`, `threshold_used`, `flagged_window`, `consecutive_flags`, `h1_nonempty`
  - PCA metadata: `pca_requested`, `pca_effective`, `pca_clamped`, `pca_disabled`, `dr_method`
- `baseline_params.json`: JSON snapshot containing:
  - `config_snapshot`: clean-baseline config and dataset metadata
  - `pca`: effective dimensionality, clamping/disabled flags, and optional explained variance ratios
  - `calibrator`: feature-wise centers/scales for the robust z-score baseline
  - `threshold`: mode, quantile, and calibrated threshold value
  - `summary`: warmup/calibration window counts, windows evaluated, flagged windows, flag rate, H1 non-empty frequency, and anomaly score percentiles (p95, p99, p99.5)

## Repository cleanup

To remove generated junk (caches, bytecode, and duplicate run artifacts) without touching core source or data, run from the project root:

```bash
python scripts/cleanup_repo.py
```

**Default behavior**:

- **Always deletes**:
  - `__pycache__/` directories
  - `.pytest_cache/` directories
  - `.ipynb_checkpoints/` directories
  - `*.pyc` files
- **Outputs pruning**:
  - By default, `--prune-outputs` is enabled. It removes **only** top-level files in `outputs/` that are exact duplicates (same filename and SHA256 hash) of files under `outputs/clean/seed_0/`.
  - No other files or directories under `outputs/` are touched.
- **Never deletes by default**:
  - `.venv/`
  - `data/` (including `data/cicids2017/`)
  - Any `.ipynb` notebook files
  - `.vscode/`

**Flags**:

- `--prune-outputs` / `--no-prune-outputs`:
  - Controls duplicate pruning between `outputs/` (top level) and `outputs/clean/seed_0/` (default: `--prune-outputs` enabled).
- `--remove-results`:
  - If set, deletes the entire `results/` directory (more destructive).
- `--remove-outputs`:
  - If set, deletes the entire `outputs/` directory, including all runs (most destructive).
- `--remove-vscode`:
  - If set, also deletes the `.vscode/` directory (by default it is preserved).
- `--root PATH`:
  - Override the detected repository root (normally you can omit this and run from `tda_code/`).

