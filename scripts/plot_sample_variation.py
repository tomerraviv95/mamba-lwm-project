import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
from pathlib import Path

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

# Apply same matplotlib styling as plot_task_scores.py
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['font.size'] = 8
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize'] = [9.5, 6.45]
mpl.rcParams['axes.titlesize'] = 28
mpl.rcParams['axes.labelsize'] = 28
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

# Configuration
MODEL_TYPES = ['transformer', 'mamba', 'raw', 'wimae', 'contrawimae']
PATCH_SIZES = [4, 6, 8]

# Load aggregated results from all combinations
results = {}
for model_type in MODEL_TYPES:
    for patch_size in PATCH_SIZES:
        results_path = os.path.join(_REPO_ROOT, f"outputs/submissions/submission_sample_variation_{model_type}_patch{patch_size}/aggregated_results.json")
        if Path(results_path).exists():
            key = f"{model_type}_patch{patch_size}"
            with open(results_path, "r") as f:
                results[key] = json.load(f)
            print(f"Loaded: {results_path}")
        else:
            print(f"Warning: {results_path} not found, skipping...")

if not results:
    print("Error: No results files found. Please run training first.")
    exit(1)

# Colors and styles for model types (matching plot_task_scores.py)
arch_colors = {
    'Transformer': '#1f77b4',
    'Mamba': '#d62728',
    'Raw Channel': '#2ca02c',
    'WiMAE': '#ff7f0e',
    'ContraWiMAE': '#9467bd'
}

arch_styles = {
    'Transformer': '-',
    'Mamba': '--',
    'Raw Channel': ':',
    'WiMAE': '-.',
    'ContraWiMAE': '-'
}

model_display_names = {
    'transformer': 'Transformer',
    'mamba': 'Mamba',
    'raw': 'Raw Channel',
    'wimae': 'WiMAE',
    'contrawimae': 'ContraWiMAE'
}

# Task names for display (excluding Channel Estimation)
# Row 1: LoS/NLoS, Beam Prediction, Legend
# Row 2: Channel Interpolation, User Localization, (blank)
task_indices = [1, 2, 3, 5]
task_positions = [0, 1, 2, 3]  # subplot positions in 2x2 grid
task_names_full = [
    "LOS/NLOS Classification",
    "Beam Prediction",
    "Channel Interpolation",
    "User Localization"
]
task_ylabels = [
    "Accuracy",
    "Accuracy",
    "NMSE (dB)",
    "Localization Error (m)"
]

# Conversion functions: stored score -> raw metric
def score_to_nmse_db(score):
    """Convert normalized score back to NMSE in dB."""
    return -20.0 * score

def score_to_loc_error(score):
    """Convert normalized score back to localization error in meters."""
    return (1.0 - score) * 100.0

# Map plot index to conversion function (None = no conversion, plot raw score)
score_converters = {
    2: score_to_nmse_db,   # Channel Interpolation
    3: score_to_loc_error,  # User Localization
}

# Get sample percentages from first available result
first_result = list(results.values())[0]
percentages = first_result["experiment_config"]["sample_percentages"]

# ============================================================================
# Generate one figure per patch size
# ============================================================================
for patch_size in PATCH_SIZES:
    # Check if any data exists for this patch size
    available_models = [m for m in MODEL_TYPES if f"{m}_patch{patch_size}" in results]
    if not available_models:
        print(f"No data for patch size {patch_size}x{patch_size}, skipping...")
        continue

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Plot each task (excluding task 4)
    for plot_idx, task_num in enumerate(task_indices):
        task_key = f"task_{task_num}"
        ax = axes[task_positions[plot_idx]]

        # Plot each model type for this patch size
        for model_type in MODEL_TYPES:
            key = f"{model_type}_patch{patch_size}"
            if key not in results:
                continue

            data = results[key]
            if task_key not in data["results_by_task"]:
                continue

            task_data = data["results_by_task"][task_key]
            display_name = model_display_names[model_type]

            # Extract scores and sample counts
            n_samples = []
            scores = []
            for pct in percentages:
                pct_str = str(pct)
                if pct_str in task_data["results"]:
                    n_samples.append(task_data["results"][pct_str]["n_samples"])
                    scores.append(task_data["results"][pct_str]["score"])

            if not scores:
                continue

            # Convert scores to raw metric if needed
            converter = score_converters.get(plot_idx)
            plot_values = [converter(s) for s in scores] if converter else scores

            ax.plot(n_samples, plot_values, 'o' + arch_styles[display_name],
                    color=arch_colors[display_name],
                    linewidth=2, markersize=8)

        ax.set_title(task_names_full[plot_idx])
        ax.set_xlabel('Number of Training Samples')
        ax.set_ylabel(task_ylabels[plot_idx])
        ax.grid(True, alpha=0.3)
        if plot_idx not in score_converters:
            ax.set_ylim([0, 1])

        # Add percentage labels on top x-axis
        if n_samples:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(n_samples)
            ax2.set_xticklabels([f"{p}%" for p in percentages[:len(n_samples)]])
            ax2.set_xlabel("Training Data %", fontsize=20, style='italic')

    # Place legend below the 4 subplots (only for 8x8 patch size)
    if patch_size == 8:
        arch_handles = [
            Line2D([0], [0], color=arch_colors['Transformer'], linestyle='-',
                   linewidth=5, marker='o', markersize=16, label='Transformer'),
            Line2D([0], [0], color=arch_colors['Mamba'], linestyle='--',
                   linewidth=5, marker='o', markersize=16, label='Mamba'),
            Line2D([0], [0], color=arch_colors['Raw Channel'], linestyle=':',
                   linewidth=5, marker='o', markersize=16, label='Raw Channel'),
            Line2D([0], [0], color=arch_colors['WiMAE'], linestyle='-.',
                   linewidth=5, marker='o', markersize=16, label='WiMAE'),
            Line2D([0], [0], color=arch_colors['ContraWiMAE'], linestyle='-',
                   linewidth=5, marker='o', markersize=16, label='ContraWiMAE')
        ]
        fig.legend(handles=arch_handles, loc='lower center', fontsize=28, handlelength=4,
                   ncol=5, bbox_to_anchor=(0.5, -0.12))

    plt.tight_layout()

    filename = os.path.join(_REPO_ROOT, f"outputs/plots/performance_vs_samples_patch{patch_size}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.show()

print("\nPlotting complete!")
print(f"\nLoaded {len(results)} result files:")
for key in sorted(results.keys()):
    print(f"  - {key}")
