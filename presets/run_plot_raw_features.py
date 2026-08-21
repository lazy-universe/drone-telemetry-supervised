import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path to load datasets
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from implement.utils.helper import get_or_preprocess_genuine_dji_flights, get_or_preprocess_hardware_spoofer

def main():
    print("Loading datasets...")
    # Load raw feature data
    # (The preprocessors automatically include the raw columns in the returned dataframe now)
    df_real = get_or_preprocess_genuine_dji_flights()
    df_spoofed = get_or_preprocess_hardware_spoofer()

    features_to_plot = ['height', 'ground_speed', 'vertical_speed', 'course']
    
    # Set premium plotting theme
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16,
        'font.family': 'sans-serif'
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = {'Real DJI': '#2b7bba', 'Spoofed': '#e74c3c'}

    for idx, feat in enumerate(features_to_plot):
        ax = axes[idx]
        
        # Clean NaNs
        real_vals = df_real[feat].dropna()
        spoofed_vals = df_spoofed[feat].dropna()

        # For course angle, use circular histogram, otherwise standard KDE / Histogram
        if feat == 'course':
            # course is [0, 360)
            sns.histplot(real_vals, bins=36, color=colors['Real DJI'], alpha=0.6, label='Real DJI', kde=True, ax=ax, stat='density')
            sns.histplot(spoofed_vals, bins=36, color=colors['Spoofed'], alpha=0.6, label='Spoofed', kde=True, ax=ax, stat='density')
            ax.set_xlim(0, 360)
            ax.set_xlabel("Course Direction (degrees)")
        else:
            # Clip outlier values for plotting visibility if needed
            p99_real = np.percentile(real_vals, 99.5) if len(real_vals) > 0 else 100
            p99_spoofed = np.percentile(spoofed_vals, 99.5) if len(spoofed_vals) > 0 else 100
            max_val = max(p99_real, p99_spoofed, 5.0)
            min_val = min(np.percentile(real_vals, 0.5), np.percentile(spoofed_vals, 0.5), -5.0)

            sns.kdeplot(real_vals, color=colors['Real DJI'], fill=True, alpha=0.3, label='Real DJI', ax=ax, linewidth=2)
            sns.kdeplot(spoofed_vals, color=colors['Spoofed'], fill=True, alpha=0.3, label='Spoofed', ax=ax, linewidth=2)
            ax.set_xlim(min_val, max_val)
            
            unit = " (m)" if feat in ['height'] else " (m/s)"
            ax.set_xlabel(feat.replace('_', ' ').title() + unit)

        ax.set_ylabel("Probability Density")
        ax.set_title(f"Distribution of {feat.replace('_', ' ').title()}")
        ax.legend()

    plt.suptitle("Comparison of Raw Non-Spatial Features: Real vs. Spoofed Classes", y=0.98, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save target plot
    output_plots_dir = PROJECT_ROOT / 'implement' / 'output' / 'plots'
    output_plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_plots_dir / 'raw_features_comparison.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"✓ Beautiful comparative plot successfully saved to: {plot_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot distribution comparison for raw non-spatial features")
    parser.parse_args()
    main()
