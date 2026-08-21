# 🛸 Supervised Drone Telemetry Spoofing Detection & Classification Framework

A modular, extensible research framework for detecting GPS/telemetry spoofing and replay attacks on UAVs (Unmanned Aerial Vehicles) using physics-informed kinematic features, classical machine learning, and deep sequence architectures.

---

## 📁 Repository Structure

```
drone-telemetry-supervised/
├── README.md                                  # Framework overview & quickstart
├── .gitignore                                 # Git rules with dataset tracking exceptions
│
├── dataset/                                   # Curated & Authentic Telemetry Datasets
│   ├── genuine_dji_flights/                   # Authentic multirotor flights (12 DJI airframes)
│   │   └── consistent_dataset/                # 38 standardized, verified physical flight logs
│   ├── hardware_spoofer/                      # Broadcast Bluetooth Remote ID telemetry (Physical ESP32 logs)
│   │   └── esp32_source_telemetry.csv
│   └── curated_flights/                       # Curated experimental flight profiles & normal baselines
│       ├── baseline_flight_[1-6].csv          # Field-modified real flight replays
│       ├── easy_flight_[1-6].csv              # Discontinuous random walks
│       ├── geometry_flight_[1-6].csv          # Smooth Bezier curves
│       ├── hard_flight_[1-6].csv              # Dynamic drift replays
│       ├── medium_flight_[1-6].csv            # Geometric shapes
│       └── normal/
│           └── normal_flight_[1-6].csv        # PX4 SITL physics normal baseline
│
├── implement/                                 # Core Implementation Package
│   ├── __init__.py                            # Package init & RAPIDS GPU acceleration loader
│   ├── dataset/                               # Ephemeral intermediate dataset caches
│   ├── output/                                # Stored metrics, models, curves, and figures
│   ├── utils/                                 # Modular processing, modeling & evaluation helpers
│   │   ├── classical_ml/                      # Point-wise classifiers & physics-based threshold baseline
│   │   │   ├── classical_models.py            # Logistic Regression, Random Forest, XGBoost definitions
│   │   │   ├── classical_train_eval.py        # Point-wise training & evaluation loops
│   │   │   └── threshold_baseline.py          # Prediction-Error (PE) thresholding baseline
│   │   ├── dataset_processing/                # Dataset ingestion, formatting & splitting
│   │   │   ├── dataset_helper.py              # Dynamic data combiner, scaler & group splitters
│   │   │   ├── dji_prep.py                    # DJI log loader, BOM cleaning & unit conversion
│   │   │   └── esp32_prep.py                  # ESP32 Remote ID frame parser & alignment
│   │   ├── deep_learning/                     # Sequential deep learning architectures
│   │   │   ├── dl_models.py                   # 1D-CNN, GRU, TCN, and CNN-GRU models
│   │   │   ├── dl_train_eval.py               # Sequence training loops, early stopping & threshold tuning
│   │   │   └── sequence_helper.py             # Vectorized sliding-window rolling sequence generator
│   │   └── helper/                            # Shared utilities, features & paths
│   │       ├── ablation_utils.py              # Feature override hooks for ablation experiments
│   │       ├── features.py                    # Physics-based kinematic features (15 engineered columns)
│   │       ├── loader.py                      # Multi-source dataset aggregator
│   │       ├── logging_helper.py              # Formatted metrics & confusion matrix loggers
│   │       └── paths.py                       # Dynamic project root & directory resolution
│   └── workflows/                             # End-to-End Orchestrated Workflows
│       ├── unified_supervised_classification.py # Unified point-wise & sequence classification pipeline
│       ├── leave_one_attack_out_workflow.py    # Standard Leave-One-Attack-Out (LOAO) workflow
│       └── leave_one_attack_out_unified.py     # Multi-model unified LOAO evaluation
│
├── notebooks/                                 # Clean, Numbered Reproducibility Notebooks
│   ├── 01_supervised_exploration.ipynb        # Exploratory Data Analysis & 2D Trajectories
│   ├── 02_quantitative_dataset_analysis.ipynb # Dataset distributions, statistics & overlaps
│   ├── 03_baselines_and_data_splits.ipynb     # Baseline classifiers across Random, Flight & Device splits
│   ├── 04_ablation_studies.ipynb              # Feature exclusion, representation & window length ablations
│   ├── 05_cross_hardware_and_subattack_eval.ipynb # Per-class recall, 5-Fold cross-device CV & artifact study
│   └── 06_robustness_and_debouncing.ipynb     # Alarm debouncing sweep & validation threshold curves
│
└── presets/                                   # Standalone CLI Entry Points
    ├── run_unified_supervised_classification.py # Main benchmark preset (supports --split-mode, --tune-thresholds)
    ├── run_leave_one_attack_out.py             # LOAO zero-shot attack generalization preset
    ├── run_seed_experiment.py                  # Multi-seed stability & standard deviation analysis
    ├── run_debounce_eval.py                    # Consecutive positive alarm debouncing filter evaluation
    ├── run_plot_val_threshold_curves.py        # Validation threshold tuning curve visualizer
    └── run_plot_raw_features.py                # Distribution comparison for non-spatial raw features
```

---

## ⚡ Key Features

* **Physics-Informed Kinematics**: Computes 15 derived physical flight features including ground/vertical speeds, accelerations, jerks, turn rates, path curvature, bearing entropy, prediction errors ($PE$), and motion smoothness ($MS$).
* **Dual Evaluation Paradigms**:
  * **Point-Wise Models**: Logistic Regression, Random Forest, XGBoost, and Physics Thresholding ($PE$).
  * **Sequence Models**: 1D-CNN, GRU, Temporal Convolutional Networks (TCN), and Hybrid CNN-GRU.
* **Rigorous Validation Splits**:
  * **Random Split**: Shuffled row-level baseline.
  * **Flight-Wise Split**: Strict separation of flight logs (prevents temporal leakage).
  * **Device-Wise Split**: Strict cross-hardware holdout across drone models (evaluates generalization to unseen drone hardware).
  * **Leave-One-Attack-Out (LOAO)**: Holdout of entire attack trajectory patterns to measure zero-shot spoofer detection.
* **Operational Defense & Benchmarks**:
  * **Consecutive Alarm Debouncing**: Evaluates alert thresholding ($K \in [1, 20]$) to suppress false positives caused by momentary network packet drops.
* **GPU Acceleration**: Transparent CUDA acceleration with automatic PyTorch GPU device allocation and optional RAPIDS `cuml.accel` bindings.

---

## 🚀 Quickstart & CLI Presets

Execute presets directly from the project root:

```bash
# 1. Run Unified Supervised Classification across all models & splits
python3 presets/run_unified_supervised_classification.py --split-mode all --epochs 15

# 2. Run with Validation Threshold Tuning
python3 presets/run_unified_supervised_classification.py --split-mode flight --tune-thresholds

# 3. Evaluate Leave-One-Attack-Out (LOAO) Generalization
python3 presets/run_leave_one_attack_out.py --model-type all

# 4. Run Multi-Seed Stability Sweep (5 seeds)
python3 presets/run_seed_experiment.py --split flight --n-seeds 5

# 5. Evaluate Alarm Debouncing Filter
python3 presets/run_debounce_eval.py --split flight --mode baseline

# 6. Plot Non-Spatial Feature Distributions
python3 presets/run_plot_raw_features.py
```

---

## 📓 Interactive Notebooks

All Jupyter notebooks are located in [`notebooks/`](./notebooks/) and are self-contained with relative path resolution:

| Notebook | Purpose |
| :--- | :--- |
| **`01_supervised_exploration.ipynb`** | EDA, correlation heatmaps, feature separability, and 2D spatial trajectory visualizations. |
| **`02_quantitative_dataset_analysis.ipynb`** | Comprehensive statistical breakdown across DJI hardware, ESP32 broadcasts, and simulated trajectories. |
| **`03_baselines_and_data_splits.ipynb`** | End-to-end execution of baseline classifiers, threshold tuning, LOAO, and 5-Fold Cross Validation. |
| **`04_ablation_studies.ipynb`** | Feature ablation sweeps, raw vs engineered representation comparisons, kinematic subsets, and window length sweeps ($T \in [5, 50]$). |
| **`05_cross_hardware_and_subattack_eval.ipynb`** | Sub-attack per-class recall analysis, cross-hardware 5-Fold CV, and motion smoothness artifact study. |
| **`06_robustness_and_debouncing.ipynb`** | Operational alarm debouncing sweeps and validation threshold curves. |
