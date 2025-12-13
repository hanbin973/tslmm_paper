import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys

def plot_benchmark_with_errorbars(csv_path, output_path):
    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find '{csv_path}'.")
        sys.exit(1)

    # 2. Calculate Standard Error of the Mean (SEM)
    # SEM = Standard Deviation / sqrt(N)
    df["ts_sem"] = df["ts_std_sec"] / np.sqrt(df["iterations"])
    df["arg_sem"] = df["arg_std_sec"] / np.sqrt(df["iterations"])

    # 3. Create Categorical Labels for Mu (for consistent coloring)
    unique_mus = sorted(df["mu"].unique())
    # Create a palette with enough colors
    mu_palette = sns.color_palette("viridis", n_colors=len(unique_mus))
    # Map each specific mu float value to a color
    mu_color_map = {mu: color for mu, color in zip(unique_mus, mu_palette)}

    # 4. Setup Plot Grid
    sns.set_theme(style="whitegrid", context="paper") # 'paper' context is often better for saved files
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Panel A: Runtime Performance with SEM Error Bars ---
    ax_runtime = axes[0]
    
    # We loop manually to have full control over error bars + categorical colors
    for mu in unique_mus:
        subset = df[df["mu"] == mu].sort_values("n_samples")
        label_str = f"{mu:.0e}" # e.g., "1e-08"
        color = mu_color_map[mu]
        
        # Plot TSKit (Solid Line, Circle Marker)
        ax_runtime.errorbar(
            subset["n_samples"], 
            subset["ts_mean_sec"], 
            yerr=subset["ts_sem"], 
            label=rf"TSKit (Branch) | $\mu$={label_str}",
            fmt='-o', color=color, capsize=4, linewidth=2, markersize=6, alpha=0.9
        )
        
        # Plot ARG-Needle (Dashed Line, Square Marker)
        ax_runtime.errorbar(
            subset["n_samples"], 
            subset["arg_mean_sec"], 
            yerr=subset["arg_sem"], 
            label=rf"ARG (MatMul) | $\mu$={label_str}",
            fmt='--s', color=color, capsize=4, linewidth=2, markersize=6, alpha=0.9
        )

    ax_runtime.set_xscale("log")
    ax_runtime.set_yscale("log")
    ax_runtime.set_title(r"Runtime Comparison ($\pm$Std. Error)", fontsize=14, fontweight='bold')
    ax_runtime.set_ylabel("Execution Time (s)", fontsize=12)
    ax_runtime.set_xlabel("Sample Size (haploid)", fontsize=12)
    
    # Custom Legend
    ax_runtime.legend(title="Method | Mutation Rate", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)

    # --- Panel B: Correlation ---
    ax_corr = axes[1]
    
    # Convert mu to string for categorical coloring in Seaborn
    df["mu_cat"] = df["mu"].apply(lambda x: f"{x:.0e}")
    
    sns.lineplot(
        data=df,
        x="n_samples",
        y="correlation",
        hue="mu_cat",      # Categorical Coloring
        palette="viridis", # Matches the manual palette in Panel A
        marker="o",
        ax=ax_corr,
        linewidth=2.5,
        markersize=9
    )

    ax_corr.set_title("Method Agreement (Pearson Correlation)", fontsize=14, fontweight='bold')
    ax_corr.set_ylabel("Correlation (r)", fontsize=12)
    ax_corr.set_xlabel("Sample Size (haploid)", fontsize=12)
    
    # Zoom in on high correlation area
    min_corr = df["correlation"].min()
    ax_corr.set_ylim(max(0.0, min_corr - 0.01), 1.002)
    ax_corr.legend(title=r"Mutation Rate ($\mu$)", bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    
    # 5. Save Plot
    print(f"Saving plot to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Benchmark Results with Error Bars")
    parser.add_argument("--input", type=str, required=True, help="Path to aggregated summary CSV")
    parser.add_argument("--output", type=str, required=True, help="Path to output PNG image")
    
    args = parser.parse_args()
    
    plot_benchmark_with_errorbars(args.input, args.output)