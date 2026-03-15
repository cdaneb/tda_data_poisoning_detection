# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Research framework for detecting data poisoning attacks in online ML systems using Topological Data Analysis (TDA) and persistent homology. Combines streaming SGD learning with sliding-window Vietoris-Rips persistent homology, PCA dimensionality reduction, and robust calibration (median/MAD).

The core hypothesis: poisoned samples distort the geometric structure of the data manifold, and these distortions are detectable via shifts in topological signatures (persistent homology) computed over sliding windows — without inspecting individual samples.

---

## Commands

### Setup
```bash
source .venv/Scripts/activate   # Windows bash
pip install -r requirements.txt
```

### Tests
```bash
python -m pytest -q                                  # full suite
python -m pytest tests/test_tda_monitor.py -v        # single module
python -m pytest -k "test_name" -v                   # single test
```

### Run Experiments
```bash
# Synthetic poisoning experiment
python -m src.experiments.run_experiment \
  --n_steps 3000 --poison_rate 0.5 \
  --poison_mode label_flip --point_cloud_mode residuals

# Clean baseline on CICIDS2017
python -m src.experiments.run_clean_baseline \
  --seed 0 --day Monday --max-rows 20000
```

Outputs go to `outputs/` (gitignored). Each run writes `window_metrics.csv`, `baseline_params.json`, `config.json`, and plots. Use **`--output-dir`** to avoid overwriting runs (e.g. `--output-dir outputs/clean_tuesday` for Tuesday; keep Monday at `outputs/clean/seed_0`). Prefer **`--score-from h1_extended`** for poisoning experiments.

---

## Leakage Rules — Never Violate

These invariants are fundamental to the research validity. Any code that touches the pipeline must respect them:

- **Scaler**: fit ONLY on the initial warmup prefix of the real data stream. Never refit or update after warmup ends.
- **PCA**: fit ONLY during the warmup phase using warmup window point clouds. Frozen for all subsequent windows.
- **Threshold**: derived ONLY from calibration-phase anomaly scores via empirical quantile. Never adjusted after calibration ends.
- **Data ordering**: never shuffle the stream. Temporal order is essential — the experiment models a real streaming scenario.
- **Baseline statistics (median/MAD)**: computed ONLY from calibration-phase scores. Never updated during detection.

Violating any of these introduces data leakage and invalidates experimental results.

---

## Architecture

The pipeline layers from bottom to top:

1. **Core TDA** (`src/homology.py`, `src/summaries.py`) — ripser wrapper + H0/H1 summary statistics (max_persistence, persistence_count, persistence_entropy)
2. **Data streaming** (`src/streaming/stream.py`, `window_buffer.py`) — `DataStream` iterator over `(t, x, y)` tuples; `WindowBuffer` ring buffer storing the last W samples
3. **Online learning** (`src/streaming/online_model.py`) — `OnlineLearner` wraps SGDClassifier with `partial_fit()`
4. **TDA monitoring** (`src/streaming/tda_monitor.py`, `baseline.py`) — `TDAMonitor` orchestrates the full detection loop; `BaselineCalibrator` handles robust z-score normalization
5. **Experiments** (`src/experiments/`) — `run_experiment.py` for synthetic poisoning; `run_clean_baseline.py` for clean validation

### Key Data Flow (per streaming step)
```
DataStream → OnlineLearner.update() → WindowBuffer.add()
  → [every stride steps] TDAMonitor.update():
      WindowBuffer.get_point_cloud()
      → PCA (lazy-fit during warmup, frozen thereafter)
      → ripser persistent homology
      → feature extraction → robust z-score → L2 norm score
      → empirical quantile threshold → consecutive-flag detection
```

### TDAMonitor Phases
- **Warmup** (first 50 windows): accumulate PCA training samples; model stabilization; no scoring
- **Calibration** (next 50 windows): fit PCA; compute and store anomaly scores; derive threshold at `threshold_quantile`
- **Detection** (all remaining windows): threshold and PCA frozen; flag if k ≥ 2 consecutive windows exceed threshold

### Key Hyperparameters
```python
window_size       = 50       # samples per sliding window
stride            = 10       # steps between window evaluations
pca_components    = 15       # dimensions after reduction
warmup_windows    = 50       # windows used for model/PCA stabilization
calibration_windows = 50     # windows used to fit threshold
threshold_quantile = 0.995   # empirical quantile for anomaly threshold
k_consecutive     = 2        # consecutive anomalous windows to raise alert
point_cloud_mode  = "residuals"  # default: features + prob + error
```

### Point Cloud Modes
- `"features"` — raw feature matrix from buffer
- `"residuals"` — features + prediction probability + prediction error (appended columns); embeds model behavior into geometry; **preferred mode**

### Topological Features (12 per window)
Six features per dimension (H0 and H1). See **New topological features** below for the full list and scoring usage.

---

## New topological features

Computed in `src/summaries.py` (no giotto-tda; from raw ripser diagrams):

- **wasserstein_amplitude** — L1 Wasserstein distance from the empty diagram (= sum of persistences).
- **landscape_amplitude** — L2 norm of the first persistence landscape (discretized filtration grid from min birth to max death).
- **betti_curve_mean** — Mean of the Betti curve over the filtration range.

All **12 features** (6 H0 + 6 H1) are written to CSVs for logging and analysis. Only a subset is used for anomaly scoring depending on `score_from` (see **h1_extended scoring mode**).

---

## h1_extended scoring mode

- **Mode:** `score_from="h1_extended"` in `TDAMonitor` / config.
- **H1 (when non-empty):** indices `[6,7,8,9]` — max_persistence, count, entropy, **wasserstein_amplitude**.
- **H0 fallback (when H1 empty):** indices `[0,1,2,3]` — same four features for H0.
- **Preferred scorer:** Use `--score-from h1_extended` for poisoning experiments; it dominates h1_then_h0 across the trigger matrix.
- **Clean baseline with h1_extended (Monday):** 13 flagged, FPR **0.82%**, threshold **~6.12**. Same leakage rules apply; calibration uses first 50+50 windows.

---

## Poisoning Modes (`src/streaming/poison.py`)

### Fixed Bugs (do not reintroduce)
- `run_experiment.py` previously used `warmup_windows=5` and `threshold=3.0`
  (hardcoded). Both are fixed. Never hardcode a threshold in any experiment path.
- `_shade_poison()` previously drew the poison window region regardless of
  `poison_rate`. Fixed: shading only draws when `poison_rate > 0.0`.
- `run_clean_baseline.py` previously passed `threshold=3.0` to `TDAMonitor`
  directly. Fixed: threshold is now always derived from empirical quantile
  calibration. The `threshold=0.0` placeholder is intentional and unused.

### Currently Implemented
- **`label_flip`** — randomly flip labels for target class(es) at configured rate; simplest attack, tests baseline sensitivity
- **`trigger`** — add a fixed feature offset to selected dimensions and relabel; simulates backdoor-style attacks

> ⚠️ Note: `trigger` poisoning is partially implemented. Before extending it, verify the current offset/relabeling logic is correct and document the intended trigger pattern in the experiment config.

### Not Yet Implemented
- **`distributional`** — shift the feature distribution of incoming samples without label manipulation; intended to test detection of covariate shift attacks

---

## Validated Baseline Results

Status: RE-VALIDATED after bug fixes on [today's date]. Config now serializes
`threshold_mode=empirical_quantile`. This is the canonical checkpoint —
commit hash should be tagged `clean-baseline-validated`.

These are the ground-truth numbers from the clean baseline run (Monday CICIDS2017, seed=0, max_rows=20000). Use these to verify that pipeline changes have not introduced regressions:

```
Dataset:              CICIDS Monday slice
windows_evaluated:    1593
flagged_windows:      12
false_positive_rate:  ~0.75%   (expected ~0.5%, within acceptable range)
baseline_threshold:   ~6.12
balanced_accuracy:    ~1.0
H1_nonempty_freq:     ~0.83    (topological features are non-degenerate)
```

If a clean baseline run deviates materially from these numbers, investigate before proceeding. Common causes: leakage violations, scaler/PCA refitting, data shuffling, or changed hyperparameters.

---

## Trigger experiment results (full matrix)

Consolidated results for all 6 trigger configs × both scorers (h1_then_h0 vs h1_extended). **h1_extended dominates in every config** (same or better detection_delay, higher detection_rate, larger post−pre score).

### trigger_value = 3.0

| trigger_dims   | scorer       | detection_delay | detection_rate | fpr_pre_onset | post−pre score |
|----------------|-------------|-----------------|----------------|---------------|----------------|
| [0]            | h1_then_h0  | not detected    | 0.0000         | 0.0101        | 0.0841         |
| [0]            | h1_extended | 1145            | 0.0008         | 0.0152        | 0.1308         |
| [0,1,2]        | h1_then_h0  | 454             | 0.0008         | 0.0101        | 0.2556         |
| [0,1,2]        | h1_extended | 454             | 0.0017         | 0.0152        | 0.3755         |
| [0,1,2,3,4]    | h1_then_h0  | 26              | 0.0025         | 0.0101        | 0.4892         |
| [0,1,2,3,4]    | h1_extended | 26              | 0.0117         | 0.0152        | 0.8083         |

### trigger_value = 5.0

| trigger_dims   | scorer       | detection_delay | detection_rate | fpr_pre_onset | post−pre score |
|----------------|-------------|-----------------|----------------|---------------|----------------|
| [0]            | h1_then_h0  | 186             | 0.0017         | 0.0101        | 0.4722         |
| [0]            | h1_extended | 154             | 0.0125         | 0.0152        | 0.7596         |
| [0,1,2]        | h1_then_h0  | 26              | 0.0033         | 0.0101        | 0.5510         |
| [0,1,2]        | h1_extended | 22              | 0.0225         | 0.0152        | 0.9363         |
| [0,1,2,3,4]    | h1_then_h0  | 185             | 0.0075         | 0.0101        | 0.7296         |
| [0,1,2,3,4]    | h1_extended | 22              | 0.0384         | 0.0152        | 1.2552         |

---

## Tuesday CICIDS findings

- **File:** `Tuesday-WorkingHours.pcap_ISCX.csv` (loader handles it via `file_glob="Tuesday"`).
- **Label distribution (max_rows=20000):** ~93.4% benign, ~6.6% attack (FTP-Patator in first 20k rows; SSH-Patator may appear later).
- **Attack onset:** First attack in the **training stream** (time-ordered) at **index 11,333** (0-based). Use `poison_start_t=11333` (or slightly later) for Tuesday label-flip experiments.
- **Key finding:** Tuesday clean baseline with same hyperparameters as Monday yields **FPR 6.9%** (110 flagged, threshold ~494). Real attack traffic inflates scores during and after calibration, so **Monday hyperparameters do not transfer directly to Tuesday**. Calibration is still fit only on the first 100 windows, but the stream contains attack traffic later, and the score distribution is much heavier.

---

## Next Phase: Poisoning Experiments (updated priority)

The clean baseline is validated on Monday. Trigger matrix is complete with h1_extended preferred.

### Next steps (priority order)

1. **Fix Tuesday baseline** — Either (a) use only the pre-attack prefix for calibration (e.g. cap calibration so it uses only windows before stream index 11,333), or (b) run Tuesday with `max_rows` capped to the pre-attack region for a clean baseline, then introduce poisoning separately. Goal: stable Tuesday FPR in ~0.5–2% before poisoning experiments.
2. **Label-flip on Tuesday** — Run with `poison_start_t=11333` once the Tuesday baseline is stable.
3. **Output directory discipline** — Use `--output-dir` explicitly to avoid overwriting the Monday clean baseline (e.g. `--output-dir outputs/clean_tuesday` for Tuesday, keep Monday at `outputs/clean/seed_0`).

### Other planned work

- **Distributional poisoning** — not yet implemented; requires new poisoning mode in `poison.py`.
- **Label-flip** — systematic evaluation across poison rates and onset windows (Monday done; Tuesday after baseline fix).

### Evaluation Metrics (to be implemented or verified)
- **detection_delay** — windows between poison onset and first alert
- **detection_rate** — fraction of poisoned windows correctly flagged
- **fpr_stability** — false positive rate in clean regions before/after poisoning
- **model_accuracy_degradation** — drop in balanced accuracy during poisoning vs. clean baseline

### Experiment Design Principles
- Each attack type should be tested with a controlled onset: clean stream → poison begins at a known window → optionally returns to clean
- Poison rate and onset window should be config parameters, not hardcoded
- All poisoning experiments must record the ground-truth poison window indices for evaluation
- Results must be reproducible via `--seed`

---

## Configuration Pattern

All experiments use frozen dataclasses (`ExperimentConfig`, `CleanBaselineConfig`) that serialize to `config.json` alongside outputs for full reproducibility. When adding new experiment types, follow this pattern — do not use mutable config objects or hardcoded parameters.

---

## CICIDS2017 Data

Raw CSVs live in `data/` (gitignored). `src/streaming/cicids2017.py` loads them: drops non-numeric columns, removes NaN/inf rows, binarizes labels (0=benign, 1=attack).

- **Monday:** `Monday-WorkingHours.pcap_ISCX.csv` — ~20k rows, ~77 features, nearly all benign; canonical clean baseline and poisoning experiments.
- **Tuesday:** `Tuesday-WorkingHours.pcap_ISCX.csv` — contains FTP-Patator (and SSH-Patator later in file). See **Tuesday CICIDS findings** for label distribution and attack onset; Monday hyperparameters do not transfer (Tuesday baseline FPR ~6.9% with current settings).
