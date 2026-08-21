import sys
import argparse
from pathlib import Path

# Add project root directory to Python path for importing implement modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from implement.workflows.leave_one_attack_out_unified import run_leave_one_attack_out_workflow
from implement.utils.helper.ablation_utils import (
    apply_ablation_overrides, get_ablation_suffix, DEFAULT_SEED
)


def main():
    parser = argparse.ArgumentParser(
        description="Workflow Preset: Unified Leave-One-Attack-Out Telemetry Classification. "
                    "Note: This workflow strictly evaluates normal flights using a flight-wise split to prevent trajectory leakage."
    )
    parser.add_argument(
        "--models",
        choices=["all", "point", "sequence"],
        default="all",
        help="Choose pointwise models, sequence models, or both to evaluate (default: all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible partitioning and initialisation (default: 42)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of epochs for sequence models (default: 15)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for sequence models (default: 64)"
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

    models_to_run = ["point", "sequence"] if args.models == "all" else [args.models]

    for model_type in models_to_run:
        print(f"\n=======================================================")
        print(f"=== Running Leave-One-Attack-Out for {model_type.upper()} models ===")
        print(f"=======================================================")
        run_leave_one_attack_out_workflow(
            model_type=model_type,
            random_state=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_suffix=ablation_suffix
        )

if __name__ == "__main__":
    main()
