import sys
import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from implement.utils.helper.ablation_utils import (
    apply_ablation_overrides, get_ablation_suffix, DEFAULT_SEED
)


def main():
    parser = argparse.ArgumentParser(description="Unified supervised classification preset")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs for sequence models (default: 15)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for sequence models")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--suppress", action="store_true", help="Skip image generation while keeping console logs")
    parser.add_argument(
        "--split-mode",
        choices=["all", "flight", "device", "random"],
        default="all",
        help="Run all split modes or only one mode",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Perform K-fold cross-validation (only supports split-mode=flight)",
    )
    parser.add_argument(
        "--models",
        choices=["all", "point", "sequence"],
        default="all",
        help="Choose pointwise models, sequence models, or both",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["engineered", "raw"],
        default="engineered",
        help="Feature mode: engineered or raw features",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform threshold tuning on validation set (skips retraining for deep learning models)",
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

    # Enforce split-mode=flight if cv is requested
    if args.cv:
        print("[Notice] Cross-validation requested. Restricting evaluation to split-mode=flight only.")
        if args.split_mode != "flight":
            if args.split_mode != "all":
                print(f"[Warning] Cross-validation only supports flight-wise split. Overriding split-mode from '{args.split_mode}' to 'flight'.")
            args.split_mode = "flight"

    ablation_suffix = get_ablation_suffix(args.window_size, args.exclude_features)
    apply_ablation_overrides(args.window_size, args.exclude_features, PROJECT_ROOT)

    # Import the workflow after variables are modified
    from implement.workflows.unified_supervised_classification import run_unified_supervised_workflow

    run_unified_supervised_workflow(
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_state=args.seed,
        suppress=args.suppress,
        split_mode=args.split_mode,
        model_scope=args.models,
        feature_mode=args.feature_mode,
        tune=args.tune,
        output_suffix=ablation_suffix,
        cv=args.cv
    )


if __name__ == "__main__":
    main()
