import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root directory to Python path for importing implement modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from implement.workflows.unified_supervised_classification import run_unified_supervised_workflow

from implement.utils.helper.ablation_utils import (
    apply_ablation_overrides, get_ablation_suffix, DEFAULT_SEED
)


def main():
    parser = argparse.ArgumentParser(description="Preset: Run Unified Supervised Classification across Multiple Seeds")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs for sequence models (default: 15)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for sequence models (default: 64)")
    parser.add_argument(
        "--split-mode",
        choices=["all", "flight", "device", "random"],
        default="all",
        help="Split mode to run (default: all)",
    )
    parser.add_argument(
        "--models",
        choices=["all", "point", "sequence"],
        default="all",
        help="Choose pointwise models, sequence models, or both (default: all)",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["engineered", "raw"],
        default="engineered",
        help="Feature mode: engineered or raw features (default: engineered)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform threshold tuning on validation set for pointwise and sequence models",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="17,42,137,521,2026",
        help="Comma-separated list of random seeds to evaluate (default: 17,42,137,521,2026)"
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

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    print(f"Seeds to evaluate: {seeds}")

    # Accumulate results
    all_runs = []

    for seed in seeds:
        print(f"\n{'='*75}")
        print(f"=== RUNNING SEED: {seed} ===")
        print(f"{'='*75}")
        
        # Suppress image plotting during multi-seed evaluation
        summary = run_unified_supervised_workflow(
            epochs=args.epochs,
            batch_size=args.batch_size,
            random_state=seed,
            suppress=True,
            split_mode=args.split_mode,
            model_scope=args.models,
            feature_mode=args.feature_mode,
            tune=args.tune,
            output_suffix=f"_seed_{seed}" + ablation_suffix,
        )
        
        # summary has structure: {split_mode: {model_name: {metric_name: value}}}
        for split, models_dict in summary.items():
            for model_name, metrics in models_dict.items():
                row = {
                    "Seed": seed,
                    "Split": split,
                    "Model": model_name,
                }
                row.update(metrics)
                all_runs.append(row)

    if not all_runs:
        print("No runs completed successfully.")
        return

    df = pd.DataFrame(all_runs)
    
    # Save raw runs
    output_dir = PROJECT_ROOT / "implement" / "output" / args.feature_mode / "multi_seeds"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "multi_seed_runs.csv", index=False)
    
    # Compile statistics
    # Group by Split, Model and calculate Mean and Std
    metrics_cols = [col for col in df.columns if col not in ["Seed", "Split", "Model", "confusion_matrix"]]
    
    summary_rows = []
    grouped = df.groupby(["Split", "Model"])
    for (split, model), group in grouped:
        summary_row = {
            "Split": split,
            "Model": model,
            "Runs": len(group)
        }
        for col in metrics_cols:
            mean_val = group[col].mean()
            std_val = group[col].std()
            summary_row[f"{col}_Mean_%"] = round(mean_val * 100, 2) if mean_val <= 1.0 else round(mean_val, 2)
            summary_row[f"{col}_Std_%"] = round(std_val * 100, 2) if std_val <= 1.0 else round(std_val, 2)
        summary_rows.append(summary_row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(output_dir / "multi_seed_summary.csv", index=False)
    
    print(f"\n{'='*75}")
    print("=== MULTI-SEED STATISTICAL SUMMARY ===")
    print(f"{'='*75}")
    print(df_summary.to_string(index=False))
    print(f"\n✓ Saved raw runs to: {output_dir / 'multi_seed_runs.csv'}")
    print(f"✓ Saved stats summary to: {output_dir / 'multi_seed_summary.csv'}")

if __name__ == "__main__":
    main()
