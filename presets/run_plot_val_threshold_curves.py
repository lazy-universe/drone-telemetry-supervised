"""
Preset: Plot Validation/Test Threshold vs F1-Score Curves
=========================================================
Recreates the exact validation/test dataset split and plots the F1-Score vs. 
Decision Threshold curves for sequence or pointwise models.

Usage:
    python run_plot_val_threshold_curves.py --split flight --feature-mode engineered --model-type sequence
    python run_plot_val_threshold_curves.py --split flight --feature-mode engineered --model-type pointwise
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import joblib
from pathlib import Path
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from implement.utils.helper import get_or_preprocess_genuine_dji_flights, get_or_preprocess_hardware_spoofer, SUPERVISED_FEATURES
from implement.utils.deep_learning.sequence_helper import prepare_sequence_dataset
from implement.utils.deep_learning.dl_models import (
    CNNClassifier, GRUClassifier, TCNClassifier, CNNGRUClassifier
)
from implement.utils.dataset_processing.dataset_helper import combine_and_split_supervised_data
from implement.utils.helper.ablation_utils import (
    apply_ablation_overrides, get_ablation_suffix, DEFAULT_SEED
)

RAW_FEATURES = ['height', 'ground_speed', 'vertical_speed', 'course']


def main():
    parser = argparse.ArgumentParser(description="Preset: Plot Threshold vs F1-Score Curves")
    parser.add_argument(
        "--split",
        choices=["flight", "device"],
        default="flight",
        help="Which split mode was evaluated (default: flight)"
    )
    parser.add_argument(
        "--feature-mode",
        choices=["engineered", "raw"],
        default="engineered",
        help="Feature mode: engineered or raw features (default: engineered)"
    )
    parser.add_argument(
        "--model-type",
        choices=["sequence", "pointwise"],
        default="sequence",
        help="Type of models to evaluate: sequence or pointwise (default: sequence)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used in training (default: 42)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Explicit path to the directory containing the model weights"
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

    # Recreate output base path matching the new partitioned directories
    output_base = PROJECT_ROOT / "implement" / "output" / args.feature_mode
    
    if args.model_type == "sequence":
        default_dir_name = f"sequence_tuned_{args.split}{ablation_suffix}"
        plot_name = "validation_threshold_vs_f1_curves.png"
        eval_set_name = "Validation Set"
    else:
        default_dir_name = f"point_baseline_{args.split}{ablation_suffix}"
        plot_name = "test_threshold_vs_f1_curves.png"
        eval_set_name = "Test Set (Pointwise has no validation set)"

    models_dir = output_base / default_dir_name / "models"
    plots_dir = output_base / default_dir_name / "plots"
    
    if args.model_dir:
        models_dir = Path(args.model_dir)
        plots_dir = models_dir.parent / "plots"
        
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_plot = plots_dir / plot_name

    print("=====================================================================")
    print("=== PLOTTER: THRESHOLD VS F1-SCORE ===")
    print(f"=== Model Type={args.model_type.upper()} | Split={args.split.upper()} | Feature Mode={args.feature_mode} ===")
    print("=====================================================================")

    # Select features based on mode
    features = RAW_FEATURES if args.feature_mode == "raw" else SUPERVISED_FEATURES

    # Recreate split
    print(f"\nLoading and splitting datasets to recreate {eval_set_name}...")
    dji_seq = get_or_preprocess_genuine_dji_flights(filter_length_100=True, features=features)
    esp32_df = get_or_preprocess_hardware_spoofer()

    if args.model_type == "sequence":
        _, _, X_eval, y_eval, _, _, _, _ = prepare_sequence_dataset(
            dji_seq, esp32_df, random_state=args.seed, features=features, split_mode=args.split
        )
    else:
        # Pointwise evaluation uses the test set since it does not have a separate validation set in the pipeline
        _, _, X_eval, y_eval, _, _ = combine_and_split_supervised_data(
            dji_seq, esp32_df, random_state=args.seed, features=features, split_mode=args.split
        )
        
    y_eval = y_eval.astype(int)
    print(f"✓ Evaluation set recreated: shape = {X_eval.shape}")

    # Set up models based on type
    if args.model_type == "sequence":
        model_specs = {
            "CNN": (CNNClassifier, "cnn_classifier.pth"),
            "GRU": (GRUClassifier, "gru_classifier.pth"),
            "TCN": (TCNClassifier, "tcn_classifier.pth"),
            "CNN-GRU": (CNNGRUClassifier, "cnn_gru_classifier.pth"),
        }
    else:
        model_specs = {
            "Logistic_Regression": "Logistic_Regression_Classifier.joblib",
            "Random_Forest": "Random_Forest_Classifier.joblib",
            "XGBoost": "XGBoost_Classifier.joblib",
        }

    plt.figure(figsize=(10, 6), dpi=150)
    thresholds = np.linspace(0.0, 1.0, 101)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plotted_any = False
    for model_name, spec in model_specs.items():
        if args.model_type == "sequence":
            model_class, file_name = spec
            pth_path = models_dir / file_name
            if not pth_path.exists():
                print(f"✗ Skipping {model_name} (file not found: {file_name})")
                continue
            
            print(f"Computing validation curve for {model_name}...")
            model = model_class(input_dim=X_eval.shape[2]).to(device)
            model.load_state_dict(torch.load(pth_path, map_location=device))
            model.eval()
            
            with torch.no_grad():
                X_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)
                logits = model(X_tensor).squeeze(-1)
                y_prob = torch.sigmoid(logits).cpu().numpy()
        else:
            file_name = spec
            joblib_path = models_dir / file_name
            if not joblib_path.exists():
                print(f"✗ Skipping {model_name} (file not found: {file_name})")
                continue
                
            print(f"Computing test curve for {model_name}...")
            model = joblib.load(joblib_path)
            
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_eval)[:, 1]
            else:
                y_prob = model.decision_function(X_eval)
                y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-5)
            
        f1_scores = []
        for t in thresholds:
            preds = (y_prob >= t).astype(int)
            f1_scores.append(f1_score(y_eval, preds, zero_division=0))
            
        f1_scores = np.array(f1_scores)
        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        # Print to terminal
        print(f"  ✓ {model_name} -> Best Threshold: {best_thresh:.3f} | Max F1: {best_f1*100:.2f}%")
        
        # Plot curve
        plt.plot(thresholds, f1_scores, label=model_name, lw=2)
        plotted_any = True

    if not plotted_any:
        print(f"\n✗ Error: No model files found in directory '{models_dir}'.")
        sys.exit(1)

    plt.xlabel("Decision Threshold", fontsize=12)
    plt.ylabel("F1-Score", fontsize=12)
    plt.title(f"{eval_set_name} Threshold vs. F1-Score (Split: {args.split.upper()})", fontsize=14, pad=15)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_plot, bbox_inches="tight")
    plt.close()
    
    print(f"\n✓ Saved F1-threshold comparison curve to:")
    print(f"  - {out_plot}")

if __name__ == "__main__":
    main()
