"""
Benchmark Script: RAM and Latency vs Patch Size for Mamba and Transformer

Measures GPU memory consumption and inference latency as a function of patch size
for both architectures using synthetic Gaussian noise data.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
import time
from typing import Dict, List, Tuple
from tqdm import tqdm
import warnings
import gc
warnings.filterwarnings("ignore")

from pretrained_model import lwm
from mamba_model import lwm_mamba
from utils import patch_maker

# Set deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Plot styling (matching task_scores_comparison.py)
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Model configurations
ELEMENT_LENGTH = 32
D_MODEL = 128
N_LAYERS = 12
N_HEADS = 8
DROPOUT = 0.1
D_STATE = 8
D_CONV = 4
EXPAND = 1.2

# Benchmark settings
BATCH_SIZE = 4
MATRIX_SIZE = 240  # Fixed channel matrix size (240x240, divisible by 4, 6, 8)
WARMUP_RUNS = 10
MEASURE_RUNS = 50

# Patch sizes to test
PATCH_SIZES = [4, 6, 8]


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_synthetic_channels(batch_size: int, n_rows: int, n_cols: int) -> np.ndarray:
    """Generate synthetic channel data using Gaussian noise."""
    real_part = np.random.randn(batch_size, n_rows, n_cols)
    imag_part = np.random.randn(batch_size, n_rows, n_cols)
    return real_part + 1j * imag_part


def tokenize_channels(channels: np.ndarray, patch_rows: int, patch_cols: int) -> torch.Tensor:
    """Tokenize channel matrices into patches."""
    patches = patch_maker(channels, patch_rows=patch_rows, patch_cols=patch_cols)

    batch_size = patches.shape[0]
    patch_size = patches.shape[2]

    # Create CLS token
    cls_token = np.ones((batch_size, 1, patch_size)) * 0.2
    tokens_with_cls = np.concatenate([cls_token, patches], axis=1)

    return torch.from_numpy(tokens_with_cls).float()


def get_gpu_memory_mb() -> float:
    """Get current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def get_peak_gpu_memory_mb() -> float:
    """Get peak GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def measure_latency_and_memory(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    warmup_runs: int = WARMUP_RUNS,
    measure_runs: int = MEASURE_RUNS
) -> Tuple[Dict[str, float], float]:
    """Measure inference latency and peak GPU memory.

    Returns:
        Tuple of (latency_dict, peak_memory_mb)
    """
    model.eval()
    tokens = tokens.to(DEVICE)

    # Reset memory stats
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(tokens)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    # Measure latency
    latencies = []
    with torch.no_grad():
        for _ in range(measure_runs):
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(tokens)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

    # Get peak memory
    peak_memory = get_peak_gpu_memory_mb()

    return {
        "mean": float(np.mean(latencies)),
        "std": float(np.std(latencies))
    }, peak_memory


def benchmark_architecture(model_type: str, patch_sizes: List[int]) -> List[Dict]:
    """Benchmark an architecture across different patch sizes."""
    results = []

    print(f"\nBenchmarking {model_type.upper()}...")

    for patch_size in tqdm(patch_sizes, desc=f"{model_type.upper()}"):
        clear_gpu_memory()

        # Generate synthetic channels
        total_samples = BATCH_SIZE * MEASURE_RUNS
        channels = generate_synthetic_channels(total_samples, MATRIX_SIZE, MATRIX_SIZE)

        # Tokenize a batch
        sample_tokens = tokenize_channels(channels[:BATCH_SIZE], patch_size, patch_size)
        actual_seq_len = sample_tokens.shape[1]
        n_patches = actual_seq_len - 1

        # Calculate max_len for model
        max_len = actual_seq_len + 10

        # Calculate element_length based on patch size
        element_length = 2 * patch_size * patch_size

        print(f"\n  Patch {patch_size}x{patch_size}: seq_len={actual_seq_len}, element_length={element_length}")

        # Initialize model
        clear_gpu_memory()

        if model_type == "transformer":
            model = lwm(
                element_length=element_length,
                d_model=D_MODEL,
                n_layers=N_LAYERS,
                max_len=max_len,
                n_heads=N_HEADS,
                dropout=DROPOUT
            ).to(DEVICE)
        else:  # mamba
            model = lwm_mamba(
                element_length=element_length,
                d_model=D_MODEL,
                n_layers=N_LAYERS,
                max_len=max_len,
                d_state=D_STATE,
                d_conv=D_CONV,
                expand=EXPAND,
                dropout=DROPOUT,
                bidirectional=True
            ).to(DEVICE)

        model.eval()

        # Measure
        try:
            latency, peak_memory = measure_latency_and_memory(model, sample_tokens)

            results.append({
                "patch_size": patch_size,
                "sequence_length": actual_seq_len,
                "n_patches": n_patches,
                "mean_latency_ms": latency["mean"],
                "std_latency_ms": latency["std"],
                "peak_memory_mb": peak_memory
            })

            print(f"    Latency: {latency['mean']:.2f} ± {latency['std']:.2f} ms")
            print(f"    Peak Memory: {peak_memory:.2f} MB")

        except RuntimeError as e:
            print(f"\n  Failed at patch_size={patch_size}: {str(e)}")

        # Cleanup
        del model, channels, sample_tokens
        clear_gpu_memory()

    return results


def plot_results(transformer_results: List[Dict], mamba_results: List[Dict]):
    """Create plots comparing latency and memory across patch sizes."""

    # Extract data
    trans_patches = [r["patch_size"] for r in transformer_results]
    trans_latency = [r["mean_latency_ms"] for r in transformer_results]
    trans_memory = [r["peak_memory_mb"] for r in transformer_results]

    mamba_patches = [r["patch_size"] for r in mamba_results]
    mamba_latency = [r["mean_latency_ms"] for r in mamba_results]
    mamba_memory = [r["peak_memory_mb"] for r in mamba_results]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Colors matching plot_task_scores.py
    transformer_color = '#1f77b4'
    mamba_color = '#d62728'

    # Plot 1: Latency
    ax1.plot(trans_patches, trans_latency, 'o-', color=transformer_color,
             linewidth=3, markersize=10)
    ax1.plot(mamba_patches, mamba_latency, 's--', color=mamba_color,
             linewidth=3, markersize=10)

    ax1.set_xlabel('Patch Size')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_xticks(PATCH_SIZES)
    ax1.set_xticklabels([f'{p}x{p}' for p in PATCH_SIZES])
    ax1.grid(True, alpha=0.3)

    # Add legend for latency plot
    style_handles_1 = [
        Line2D([0, 2], [0, 0], color=transformer_color, linestyle='-', linewidth=3, marker='o', markersize=8, label='Transformer'),
        Line2D([0, 2], [0, 0], color=mamba_color, linestyle='--', linewidth=3, marker='s', markersize=8, label='Mamba')
    ]
    ax1.legend(handles=style_handles_1, loc='upper right', handlelength=3)

    # Plot 2: Memory
    ax2.plot(trans_patches, trans_memory, 'o-', color=transformer_color,
             linewidth=3, markersize=10)
    ax2.plot(mamba_patches, mamba_memory, 's--', color=mamba_color,
             linewidth=3, markersize=10)

    ax2.set_xlabel('Patch Size')
    ax2.set_ylabel('Peak GPU Memory (MB)')
    ax2.set_xticks(PATCH_SIZES)
    ax2.set_xticklabels([f'{p}x{p}' for p in PATCH_SIZES])
    ax2.grid(True, alpha=0.3)

    # Add legend for memory plot
    style_handles_2 = [
        Line2D([0, 2], [0, 0], color=transformer_color, linestyle='-', linewidth=3, marker='o', markersize=8, label='Transformer'),
        Line2D([0, 2], [0, 0], color=mamba_color, linestyle='--', linewidth=3, marker='s', markersize=8, label='Mamba')
    ]
    ax2.legend(handles=style_handles_2, loc='upper right', handlelength=3)

    plt.tight_layout()
    plt.savefig('benchmark_patch_sizes.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nPlot saved to benchmark_patch_sizes.png")


def main():
    """Main execution."""
    set_seed(SEED)

    print("=" * 80)
    print("BENCHMARK: RAM and Latency vs Patch Size")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Matrix Size: {MATRIX_SIZE}x{MATRIX_SIZE}")
    print(f"Patch Sizes: {PATCH_SIZES}")
    print(f"Model: d_model={D_MODEL}, n_layers={N_LAYERS}")
    print("=" * 80)

    # Run benchmarks
    transformer_results = benchmark_architecture("transformer", PATCH_SIZES)
    mamba_results = benchmark_architecture("mamba", PATCH_SIZES)

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print("\nTransformer:")
    for r in transformer_results:
        print(f"  Patch {r['patch_size']}x{r['patch_size']}: "
              f"Latency={r['mean_latency_ms']:.2f}±{r['std_latency_ms']:.2f}ms, "
              f"Memory={r['peak_memory_mb']:.2f}MB")

    print("\nMamba:")
    for r in mamba_results:
        print(f"  Patch {r['patch_size']}x{r['patch_size']}: "
              f"Latency={r['mean_latency_ms']:.2f}±{r['std_latency_ms']:.2f}ms, "
              f"Memory={r['peak_memory_mb']:.2f}MB")

    # Create plot
    print("\n" + "=" * 80)
    print("Generating plot...")
    plot_results(transformer_results, mamba_results)

    print("=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
