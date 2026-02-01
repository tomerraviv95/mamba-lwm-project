import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import matplotlib as mpl


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

# Load the data
with open('submission_multi_patches_transformer/composite_score.json', 'r') as f:
    transformer_data = json.load(f)

with open('submission_multi_patches_mamba/composite_score.json', 'r') as f:
    mamba_data = json.load(f)

with open('submission_multi_patches_raw/composite_score.json', 'r') as f:
    raw_data = json.load(f)

# Extract patch sizes and convert to numeric values for plotting
def extract_patch_number(patch_name):
    """Extract the numeric patch size (e.g., 'patch_4x4' -> 4)"""
    return int(patch_name.split('_')[1].split('x')[0])

# Get all unique patch sizes across all architectures
transformer_patches = set(transformer_data['per_patch_results'].keys())
mamba_patches = set(mamba_data['per_patch_results'].keys())
raw_patches = set(raw_data['per_patch_results'].keys())
all_patches = sorted(transformer_patches | mamba_patches | raw_patches, key=extract_patch_number)
all_patches = [patch for patch in all_patches if extract_patch_number(patch) in [4, 6, 8]]

# Task names (key for JSON, display name for legend)
tasks = [
    ('LosNlosClassification', 'Los/Nlos Classification'),
    ('BeamPrediction', 'Beam Prediction'),
    ('ChannelInterpolation', 'Channel Interpolation'),
    ('ChannelEstimation', 'Channel Estimation'),
    ('ChannelCharting', 'User Localization')
]

# Colors for each architecture
arch_colors = {
    'Transformer': '#1f77b4',
    'Mamba': '#d62728',
    'Raw': '#2ca02c'
}

# Create 2x3 grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, (task_key, task_name) in enumerate(tasks):
    ax = axes[idx]

    # Extract scores for transformer
    transformer_x = []
    transformer_y = []
    for patch in all_patches:
        if patch in transformer_data['per_patch_results']:
            transformer_x.append(extract_patch_number(patch))
            transformer_y.append(transformer_data['per_patch_results'][patch]['task_scores'][task_key])

    # Extract scores for mamba
    mamba_x = []
    mamba_y = []
    for patch in all_patches:
        if patch in mamba_data['per_patch_results']:
            mamba_x.append(extract_patch_number(patch))
            mamba_y.append(mamba_data['per_patch_results'][patch]['task_scores'][task_key])

    # Extract scores for raw
    raw_x = []
    raw_y = []
    for patch in all_patches:
        if patch in raw_data['per_patch_results']:
            raw_x.append(extract_patch_number(patch))
            raw_y.append(raw_data['per_patch_results'][patch]['task_scores'][task_key])

    # Plot all architectures
    ax.plot(transformer_x, transformer_y, 'o-', color=arch_colors['Transformer'],
            linewidth=2, markersize=8)
    ax.plot(mamba_x, mamba_y, 'o--', color=arch_colors['Mamba'],
            linewidth=2, markersize=8)
    ax.plot(raw_x, raw_y, 'o:', color=arch_colors['Raw'],
            linewidth=2, markersize=8)

    ax.set_title(task_name)
    ax.set_xlabel('Patch Size')
    ax.set_ylabel('Score')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([4, 6, 8])
    ax.set_xticklabels(['4x4', '6x6', '8x8'])

# Use the 6th subplot (bottom right) for legend
ax_legend = axes[5]
ax_legend.axis('off')

# Architecture legend
arch_handles = [
    Line2D([0], [0], color=arch_colors['Transformer'], linestyle='-', linewidth=5, marker='o', markersize=16, label='Transformer'),
    Line2D([0], [0], color=arch_colors['Mamba'], linestyle='--', linewidth=5, marker='o', markersize=16, label='Mamba'),
    Line2D([0], [0], color=arch_colors['Raw'], linestyle=':', linewidth=5, marker='o', markersize=16, label='Raw Channel')
]
ax_legend.legend(handles=arch_handles, loc='center', fontsize=28, handlelength=4)

plt.tight_layout()
plt.savefig('task_scores_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("Plot saved to task_scores_comparison.png")
