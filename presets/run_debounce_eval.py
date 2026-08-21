"""
Preset: DL Sequence Model Evaluation with Debouncing
=====================================================
Evaluates pre-trained sequence (DL) models under various debounce window sizes.

Debouncing: an alert is only raised when K *consecutive* model predictions are
positive (spoofed). This post-processing step reduces transient false alarms at
the cost of slightly increased detection latency.

Strategy:
  - Load pre-trained .pth models from sequence_baseline_* or sequence_tuned_* dirs.
  - Run a single forward pass over X_test to obtain per-window probabilities.
  - Apply debounce filter at each K value in DEBOUNCE_WINDOWS.
  - Report Accuracy, Precision, Recall, F1, ROC-AUC and False-Alarm Rate per (model, K).

Usage:
    python run_debounce_eval.py --split flight
    python run_debounce_eval.py --split flight --mode tuned
    python run_debounce_eval.py --split flight --mode both

Output:
    Saved to: supervised/implement/output/debounce/debounce_<split>_<mode>.csv
"""

import sys
import argparse
import warnings
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from sklearn.exceptions import UndefinedMetricWarning
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=UndefinedMetricWarning)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

OUTPUT_BASE   = PROJECT_ROOT / "implement" / "output"
DEBOUNCE_OUT  = OUTPUT_BASE / "debounce"

# ---------------------------------------------------------------------------
# Debounce window sizes to sweep (number of consecutive positives required)
# ---------------------------------------------------------------------------
DEBOUNCE_WINDOWS = [1, 2, 3, 5, 7, 10, 15, 20]

# ---------------------------------------------------------------------------
# Sequence model registry
# ---------------------------------------------------------------------------
SEQUENCE_MODEL_FILES = {
    "cnn":           "cnn_classifier",
    "gru":           "gru_classifier",
    "tcn":           "tcn_classifier",
    "cnn_gru":       "cnn_gru_classifier",
}

SEQ_BASELINE_SPLITS = {
    "flight": OUTPUT_BASE / "sequence_baseline_flight",
    "device": OUTPUT_BASE / "sequence_baseline_device",
}
SEQ_TUNED_SPLITS = {
    "flight": OUTPUT_BASE / "sequence_tuned_flight",
    "device": OUTPUT_BASE / "sequence_tuned_device",
}


# ---------------------------------------------------------------------------
# Debounce post-processing
# ---------------------------------------------------------------------------

def apply_debounce_per_flight(y_pred_raw: np.ndarray, flight_ids: np.ndarray, k: int) -> np.ndarray:
    """
    Applies debounce logic independently for each flight to avoid false consecutive triggers
    bleeding across flight boundaries.
    """
    if k <= 1:
        return y_pred_raw.copy()

    debounced = np.zeros_like(y_pred_raw)
    unique_flights = np.unique(flight_ids)
    
    for f_id in unique_flights:
        idx = (flight_ids == f_id)
        flight_preds = y_pred_raw[idx]
        debounced[idx] = apply_debounce(flight_preds, k)
        
    return debounced


def apply_debounce(y_pred_raw: np.ndarray, k: int) -> np.ndarray:
    """
    Applies a debounce filter to a 1-D binary prediction array.

    A positive (spoofed) alert is only emitted when K or more *consecutive*
    positive predictions are observed. Once triggered the output stays 1
    until a negative prediction resets the counter.

    Args:
        y_pred_raw : np.ndarray of shape (N,) with values in {0, 1}
        k          : int — consecutive positive predictions required to trigger alert

    Returns:
        np.ndarray of shape (N,) — debounced binary predictions
    """
    if k <= 1:
        return y_pred_raw.copy()

    debounced = np.zeros_like(y_pred_raw)
    consecutive = 0
    in_alert = False

    for i, pred in enumerate(y_pred_raw):
        if pred == 1:
            consecutive += 1
        else:
            consecutive = 0
            in_alert = False

        if consecutive >= k:
            in_alert = True

        debounced[i] = 1 if in_alert else 0

    return debounced


# ---------------------------------------------------------------------------
# Model loader (with per-process cache)
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict = {}


def load_sequence_model(model_key: str, pth_path: Path,
                        n_features: int, seq_len: int,
                        device: torch.device) -> torch.nn.Module:
    """
    Loads a sequence model from disk, caches it in memory to avoid repeated
    disk reads across debounce window sweeps.
    """
    cache_key = str(pth_path)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    from implement.utils.deep_learning.dl_models import (
        CNNClassifier, GRUClassifier, TCNClassifier, CNNGRUClassifier,
    )

    model_map = {
        "cnn":           lambda: CNNClassifier(input_dim=n_features),
        "gru":           lambda: GRUClassifier(input_dim=n_features),
        "tcn":           lambda: TCNClassifier(input_dim=n_features),
        "cnn_gru":       lambda: CNNGRUClassifier(input_dim=n_features),
    }

    if model_key not in model_map:
        raise ValueError(f"Unknown model key: {model_key}")

    model_obj = model_map[model_key]()
    state = torch.load(pth_path, map_location=device)
    model_obj.load_state_dict(state)
    model_obj.to(device)
    model_obj.eval()

    _MODEL_CACHE[cache_key] = model_obj
    return model_obj


# ---------------------------------------------------------------------------
# Inference — single forward pass to get raw probabilities
# ---------------------------------------------------------------------------

def get_raw_probs(model: torch.nn.Module, X_test: np.ndarray,
                  device: torch.device, batch_size: int = 256) -> np.ndarray:
    """
    Runs model forward pass and returns sigmoid probabilities (N,).
    """
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = torch.tensor(
                X_test[i:i + batch_size], dtype=torch.float32
            ).to(device)
            out   = model(batch).squeeze(-1)
            probs = torch.sigmoid(out).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs)


# ---------------------------------------------------------------------------
# Evaluate one split directory
# ---------------------------------------------------------------------------

def evaluate_split(split_dir: Path, split_label: str,
                   debounce_windows: list, device: torch.device) -> list:
    """
    Loads all available models from split_dir, runs inference once per model,
    then applies each debounce window and records metrics.
    """
    results = []
    dataset_dir = split_dir / "dataset"
    models_dir  = split_dir / "models"

    X_test_path = dataset_dir / "X_test.npy"
    y_test_path = dataset_dir / "y_test.npy"
    flight_ids_path = dataset_dir / "flight_ids_test.npy"

    if not X_test_path.exists() or not y_test_path.exists() or not flight_ids_path.exists():
        print(f"  ✗ Test arrays (or flight_ids_test.npy) not found in {dataset_dir}")
        return results

    X_test = np.load(X_test_path)               # (N, seq_len, n_features)
    y_true = np.load(y_test_path).astype(int)   # (N,)
    flight_ids_test = np.load(flight_ids_path, allow_pickle=True)
    _, seq_len, n_features = X_test.shape

    print(f"  Test set: {len(y_true)} windows | "
          f"Positive (spoofed): {y_true.sum()} "
          f"({y_true.mean()*100:.1f}%)")

    for model_key, file_stem in SEQUENCE_MODEL_FILES.items():
        pth_path       = models_dir / f"{file_stem}.pth"
        threshold_path = models_dir / f"{file_stem}.threshold.json"

        if not pth_path.exists():
            print(f"  ✗ {model_key} — .pth not found, skipping")
            continue

        # Load best_threshold (tuned) or default 0.5 (baseline)
        threshold = 0.5
        if threshold_path.exists():
            with open(threshold_path) as f:
                threshold = json.load(f).get("best_threshold", 0.5)

        try:
            model_obj = load_sequence_model(
                model_key, pth_path, n_features, seq_len, device
            )
        except Exception as e:
            print(f"  ✗ {model_key} — load failed: {e}")
            continue

        # Single inference pass — reused across all debounce windows
        y_prob = get_raw_probs(model_obj, X_test, device)
        y_pred_base = (y_prob >= threshold).astype(int)

        print(f"  ✓ {model_key} | threshold={threshold:.4f} "
              f"| raw positives: {y_pred_base.sum()}")

        # Sweep debounce windows
        for k in debounce_windows:
            y_pred_debounced = apply_debounce_per_flight(y_pred_base, flight_ids_test, k)

            acc  = accuracy_score(y_true, y_pred_debounced)
            prec = precision_score(y_true, y_pred_debounced, zero_division=0)
            rec  = recall_score(y_true, y_pred_debounced, zero_division=0)
            f1   = f1_score(y_true, y_pred_debounced, zero_division=0)
            far  = float((y_pred_debounced[y_true == 0] == 1).sum()) / max((y_true == 0).sum(), 1)

            # ROC-AUC uses raw probs (debouncing is discrete — keep probs unchanged)
            try:
                roc = roc_auc_score(y_true, y_prob)
            except Exception:
                roc = float("nan")

            results.append({
                "Split":           split_label,
                "Model":           model_key,
                "Threshold":       round(threshold, 6),
                "Debounce_K":      k,
                "Accuracy":        round(acc * 100, 3),
                "Precision":       round(prec * 100, 3),
                "Recall":          round(rec * 100, 3),
                "F1_Score":        round(f1 * 100, 3),
                "ROC_AUC":         round(roc * 100, 3),
                "False_Alarm_Rate":round(far * 100, 3),
                "Debounced_Pos":   int(y_pred_debounced.sum()),
                "Raw_Pos":         int(y_pred_base.sum()),
                "Total_Windows":   len(y_true),
            })

        print(f"    Debounce sweep done (K={debounce_windows})")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_debounce_eval(split: str, mode: str,
                      debounce_windows: list = DEBOUNCE_WINDOWS):
    DEBOUNCE_OUT.mkdir(parents=True, exist_ok=True)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"\n[Device] Using: {device_str.upper()}")

    all_results = []
    splits_to_run = ["flight", "device"] if split == "all" else [split]
    modes_to_run  = (["baseline", "tuned"] if mode == "both"
                     else [mode])

    for sp in splits_to_run:
        for m in modes_to_run:
            split_map = SEQ_BASELINE_SPLITS if m == "baseline" else SEQ_TUNED_SPLITS
            split_dir = split_map.get(sp)

            print(f"\n{'='*65}")
            print(f"Debounce Eval — Split: {sp.upper()} | Mode: {m.upper()}")
            print(f"{'='*65}")

            if split_dir is None or not split_dir.exists():
                print(f"  ✗ Dir not found: {split_dir}")
                continue

            rows = evaluate_split(
                split_dir,
                split_label=f"{sp}_{m}",
                debounce_windows=debounce_windows,
                device=device,
            )
            all_results.extend(rows)

    if not all_results:
        print("\nNo results computed. Check model files in output directories.")
        return

    df = pd.DataFrame(all_results)

    # Pretty-print summary table
    print(f"\n{'='*65}")
    print("DEBOUNCE EVALUATION SUMMARY")
    print(f"{'='*65}")
    print(df.to_string(index=False))

    out_path = DEBOUNCE_OUT / f"debounce_{split}_{mode}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved: {out_path}")


from implement.utils.helper.ablation_utils import (
    apply_ablation_overrides, get_ablation_suffix, DEFAULT_SEED
)


def main():
    parser = argparse.ArgumentParser(
        description="Preset: DL Sequence Model Evaluation with Debouncing"
    )
    parser.add_argument(
        "--split",
        choices=["flight", "device", "all"],
        default="flight",
        help="Which data split to evaluate (default: flight)"
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "tuned", "both"],
        default="baseline",
        help="Load from sequence_baseline_* or sequence_tuned_* dirs (default: baseline)"
    )
    parser.add_argument(
        "--debounce-windows",
        type=int,
        nargs="+",
        default=DEBOUNCE_WINDOWS,
        metavar="K",
        help=(
            "Space-separated list of debounce window sizes to sweep "
            f"(default: {DEBOUNCE_WINDOWS})"
        )
    )
    parser.add_argument(
        "--feature-mode",
        choices=["engineered", "raw"],
        default="engineered",
        help="Feature mode: engineered or raw features (default: engineered)"
    )
    parser.add_argument(
        "--seq-baseline-dir",
        type=str,
        default=None,
        help="Explicit path to sequence baseline split directory"
    )
    parser.add_argument(
        "--seq-tuned-dir",
        type=str,
        default=None,
        help="Explicit path to sequence tuned split directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Explicit path for debounce output directory"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Ablation: Overwrite sequence window length (WINDOW_LEN) dynamically",
    )
    parser.add_argument(
        "--exclude-features",
        type=str,
        default=None,
        help="Ablation: Comma-separated list of features to exclude dynamically",
    )
    args = parser.parse_args()

    ablation_suffix = get_ablation_suffix(args.window_size, args.exclude_features)
    apply_ablation_overrides(args.window_size, args.exclude_features, PROJECT_ROOT)

    global OUTPUT_BASE, DEBOUNCE_OUT, SEQ_BASELINE_SPLITS, SEQ_TUNED_SPLITS
    OUTPUT_BASE = PROJECT_ROOT / "implement" / "output" / args.feature_mode
    DEBOUNCE_OUT = OUTPUT_BASE / f"debounce{ablation_suffix}"
    SEQ_BASELINE_SPLITS = {
        "flight": OUTPUT_BASE / f"sequence_baseline_flight{ablation_suffix}",
        "device": OUTPUT_BASE / f"sequence_baseline_device{ablation_suffix}",
    }
    SEQ_TUNED_SPLITS = {
        "flight": OUTPUT_BASE / f"sequence_tuned_flight{ablation_suffix}",
        "device": OUTPUT_BASE / f"sequence_tuned_device{ablation_suffix}",
    }

    if args.split == "all" and (args.seq_baseline_dir or args.seq_tuned_dir):
        print("Warning: Explicit directory flags are only applied to a single split mode at a time. Please specify --split flight or --split device.")
    else:
        if args.seq_baseline_dir:
            SEQ_BASELINE_SPLITS[args.split] = Path(args.seq_baseline_dir)
        if args.seq_tuned_dir:
            SEQ_TUNED_SPLITS[args.split] = Path(args.seq_tuned_dir)

    if args.output_dir:
        DEBOUNCE_OUT = Path(args.output_dir)

    run_debounce_eval(
        split=args.split,
        mode=args.mode,
        debounce_windows=args.debounce_windows,
    )


if __name__ == "__main__":
    main()
