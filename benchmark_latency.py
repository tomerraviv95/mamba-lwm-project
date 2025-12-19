"""
Latency Benchmark Script: Mamba vs Transformer

Measures inference latency as a function of sequence length for both architectures
using synthetic Gaussian noise data.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from pretrained_model import lwm
from mamba_model import lwm_mamba
from utils import tokenizer

# Set deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
BATCH_SIZE = 32
TARGET_SEQUENCE_LENGTHS = [16, 32, 64, 128, 256]  # Desired sequence lengths
WARMUP_RUNS = 10
MEASURE_RUNS = 100

# Patch parameters
PATCH_ROWS = 4
PATCH_COLS = 4


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_matrix_size(target_seq_len: int, patch_rows: int = PATCH_ROWS, patch_cols: int = PATCH_COLS):
    """
    Calculate matrix dimensions to achieve a target sequence length.

    Sequence length = n_patches + 1 (CLS token)
    n_patches = (n_rows / patch_rows) * (n_cols / patch_cols)

    For square matrices: n_rows = n_cols = sqrt(n_patches) * patch_size
    """
    n_patches = target_seq_len - 1  # Subtract 1 for CLS token

    # For square matrices, patches_per_side^2 = n_patches
    patches_per_side = int(np.sqrt(n_patches))

    # Calculate matrix dimension
    matrix_dim = patches_per_side * patch_rows

    return matrix_dim, matrix_dim


def generate_synthetic_channels(batch_size: int, n_rows: int, n_cols: int) -> torch.Tensor:
    """Generate synthetic channel data using Gaussian noise.

    Args:
        batch_size: Number of samples
        n_rows: Number of rows in the channel matrix
        n_cols: Number of columns in the channel matrix

    Returns:
        Complex-valued tensor of shape (batch_size, n_rows, n_cols).
    """
    # Generate real and imaginary parts
    real_part = torch.randn(batch_size, n_rows, n_cols)
    imag_part = torch.randn(batch_size, n_rows, n_cols)

    # Create complex tensor
    return torch.complex(real_part, imag_part)


def measure_latency(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    warmup_runs: int = WARMUP_RUNS,
    measure_runs: int = MEASURE_RUNS
) -> Dict[str, float]:
    """Measure inference latency with proper GPU synchronization."""
    model.eval()
    input_tensor = input_tensor.to(DEVICE)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(measure_runs):
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_tensor)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

    return {
        "mean": float(np.mean(latencies)),
        "std": float(np.std(latencies))
    }


def benchmark_model(model_type: str, target_sequence_lengths: List[int]) -> List[Dict]:
    """Benchmark a model across different sequence lengths.

    Args:
        model_type: Either 'transformer' or 'mamba'
        target_sequence_lengths: List of desired sequence lengths

    Returns:
        List of dictionaries containing benchmark results
    """
    results = []

    print(f"\nBenchmarking {model_type.upper()}...")
    for target_seq_len in tqdm(target_sequence_lengths, desc=f"{model_type.upper()}"):
        # Calculate matrix dimensions for the target sequence length
        n_rows, n_cols = calculate_matrix_size(target_seq_len)

        # Calculate what the actual max_len will be
        max_len = target_seq_len + 10  # Add some buffer

        # Initialize model
        if model_type == "transformer":
            model = lwm(
                element_length=ELEMENT_LENGTH,
                d_model=D_MODEL,
                n_layers=N_LAYERS,
                max_len=max_len,
                n_heads=N_HEADS,
                dropout=DROPOUT
            ).to(DEVICE)
        else:  # mamba
            model = lwm_mamba(
                element_length=ELEMENT_LENGTH,
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

        # Generate synthetic data and tokenize
        channels = generate_synthetic_channels(BATCH_SIZE, n_rows, n_cols)
        tokens = tokenizer(channels)
        actual_seq_len = tokens.shape[1]

        # Measure latency
        try:
            latency = measure_latency(model, tokens)
            results.append({
                "target_sequence_length": target_seq_len,
                "actual_sequence_length": actual_seq_len,
                "matrix_size": (n_rows, n_cols),
                "mean_latency": latency["mean"],
                "std_latency": latency["std"]
            })

            print(f"  Target seq_len={target_seq_len:3d}, Actual={actual_seq_len:3d}, "
                  f"Matrix={n_rows}x{n_cols}, Latency={latency['mean']:.2f}ms")

        except RuntimeError as e:
            print(f"\nFailed at target_seq_len={target_seq_len}: {str(e)}")

        # Cleanup
        del model
        torch.cuda.empty_cache()

    return results


def plot_results(transformer_results: List[Dict], mamba_results: List[Dict]):
    """Create publication-quality plot comparing latencies."""

    # Extract data
    trans_seq = [r["actual_sequence_length"] for r in transformer_results]
    trans_mean = [r["mean_latency"] for r in transformer_results]
    trans_std = [r["std_latency"] for r in transformer_results]

    mamba_seq = [r["actual_sequence_length"] for r in mamba_results]
    mamba_mean = [r["mean_latency"] for r in mamba_results]
    mamba_std = [r["std_latency"] for r in mamba_results]

    # Create figure with high quality settings
    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-darkgrid')

    # Plot with error bars
    plt.errorbar(trans_seq, trans_mean, yerr=trans_std,
                 marker='o', markersize=10, linewidth=2.5, capsize=6,
                 label='Transformer', color='#2E86AB', alpha=0.9,
                 markerfacecolor='white', markeredgewidth=2)

    plt.errorbar(mamba_seq, mamba_mean, yerr=mamba_std,
                 marker='s', markersize=10, linewidth=2.5, capsize=6,
                 label='Mamba', color='#A23B72', alpha=0.9,
                 markerfacecolor='white', markeredgewidth=2)

    # Styling
    plt.xlabel('Sequence Length', fontsize=16, fontweight='bold')
    plt.ylabel('Inference Latency (ms)', fontsize=16, fontweight='bold')
    plt.title('Inference Latency: Mamba vs Transformer\n(Batch Size = {})'.format(BATCH_SIZE),
              fontsize=18, fontweight='bold', pad=20)

    plt.legend(fontsize=14, frameon=True, shadow=True, fancybox=True,
               loc='upper left', framealpha=0.95)

    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    plt.tick_params(labelsize=12)

    # Add speedup annotations
    for i in range(min(len(trans_seq), len(mamba_seq))):
        if trans_seq[i] == mamba_seq[i]:
            speedup = trans_mean[i] / mamba_mean[i]
            mid_y = (trans_mean[i] + mamba_mean[i]) / 2
            plt.annotate(f'{speedup:.2f}x',
                        xy=(trans_seq[i], mid_y),
                        xytext=(10, 0), textcoords='offset points',
                        fontsize=10, color='#27613A', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    plt.tight_layout()

    # Save with high DPI
    output_file = 'latency_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    plt.show()


def main():
    """Main execution."""
    set_seed(SEED)

    print("=" * 70)
    print("LATENCY BENCHMARK: Mamba vs Transformer")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Sequence Lengths: {SEQUENCE_LENGTHS}")
    print(f"Model: d_model={D_MODEL}, n_layers={N_LAYERS}")
    print("=" * 70)

    # Run benchmarks
    transformer_results = benchmark_model("transformer", SEQUENCE_LENGTHS)
    mamba_results = benchmark_model("mamba", SEQUENCE_LENGTHS)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print("\nTransformer:")
    for r in transformer_results:
        print(f"  seq_len={r['sequence_length']:4d}: {r['mean_latency']:7.2f} ± {r['std_latency']:5.2f} ms")

    print("\nMamba:")
    for r in mamba_results:
        print(f"  seq_len={r['sequence_length']:4d}: {r['mean_latency']:7.2f} ± {r['std_latency']:5.2f} ms")

    print("\nSpeedup (Transformer / Mamba):")
    for i in range(min(len(transformer_results), len(mamba_results))):
        speedup = transformer_results[i]['mean_latency'] / mamba_results[i]['mean_latency']
        print(f"  seq_len={transformer_results[i]['sequence_length']:4d}: {speedup:.2f}x")

    # Create plot
    print("\n" + "=" * 70)
    print("Generating plot...")
    plot_results(transformer_results, mamba_results)

    print("=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
