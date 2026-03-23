"""
Latency Benchmark Script: Mamba vs Transformer

Measures inference latency as a function of channel matrix size for both architectures
using synthetic Gaussian noise data.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import Dict, List
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from pretrained_model import lwm
from mamba_model import lwm_mamba
from utils import patch_maker

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
BATCH_SIZE = 4  # Small batch size to avoid OOM
MATRIX_SIZES = [16, 32, 64, 128, 256]  # Channel matrix dimensions (n_rows = n_cols)
WARMUP_RUNS = 10
MEASURE_RUNS = 100  # Number of batches to average over

# Patch parameters (from train_lwm.py)
PATCH_ROWS = 4
PATCH_COLS = 4


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_synthetic_channels(batch_size: int, n_rows: int, n_cols: int) -> np.ndarray:
    """Generate synthetic channel data using Gaussian noise.

    Args:
        batch_size: Number of samples
        n_rows: Number of rows in the channel matrix
        n_cols: Number of columns in the channel matrix

    Returns:
        Complex-valued numpy array of shape (batch_size, n_rows, n_cols).
    """
    # Generate real and imaginary parts
    real_part = np.random.randn(batch_size, n_rows, n_cols)
    imag_part = np.random.randn(batch_size, n_rows, n_cols)

    # Create complex array
    return real_part + 1j * imag_part


def tokenize_channels(channels: np.ndarray) -> torch.Tensor:
    """
    Tokenize channel matrices into patches following train_lwm.py processing.

    Args:
        channels: Complex numpy array of shape (batch_size, n_rows, n_cols)

    Returns:
        Torch tensor of tokenized patches with shape (batch_size, n_patches, patch_dim)
    """
    # Use patch_maker from utils (same as train_lwm.py)
    patches = patch_maker(channels, patch_rows=PATCH_ROWS, patch_cols=PATCH_COLS)

    # Add CLS token
    batch_size = patches.shape[0]
    patch_size = patches.shape[2]

    # Create CLS token (0.2 * ones, same as in utils.py tokenizer)
    cls_token = np.ones((batch_size, 1, patch_size)) * 0.2

    # Concatenate CLS token with patches
    tokens_with_cls = np.concatenate([cls_token, patches], axis=1)

    # Convert to torch tensor
    return torch.from_numpy(tokens_with_cls).float()


def measure_latency(
    model: torch.nn.Module,
    channels: np.ndarray,
    warmup_runs: int = WARMUP_RUNS,
    measure_runs: int = MEASURE_RUNS
) -> Dict[str, float]:
    """Measure inference latency with proper GPU synchronization.

    Args:
        model: Model to benchmark
        channels: Complex channel data to generate tokens from
        warmup_runs: Number of warmup iterations
        measure_runs: Number of measurement batches

    Returns:
        Dictionary with latency statistics
    """
    model.eval()

    # Generate warmup tokens
    warmup_tokens = tokenize_channels(channels[:BATCH_SIZE]).to(DEVICE)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(warmup_tokens)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    del warmup_tokens

    # Measure across multiple batches
    latencies = []
    with torch.no_grad():
        for i in range(measure_runs):
            # Generate fresh tokens for each batch to avoid memory buildup
            batch_start = (i * BATCH_SIZE) % len(channels)
            batch_end = batch_start + BATCH_SIZE
            if batch_end > len(channels):
                # Wrap around if needed
                batch_channels = np.concatenate([
                    channels[batch_start:],
                    channels[:batch_end - len(channels)]
                ])
            else:
                batch_channels = channels[batch_start:batch_end]

            tokens = tokenize_channels(batch_channels).to(DEVICE)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(tokens)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

            # Clean up
            del tokens
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    return {
        "mean": float(np.mean(latencies)),
        "std": float(np.std(latencies))
    }


def benchmark_model(model_type: str, matrix_sizes: List[int]) -> List[Dict]:
    """Benchmark a model across different channel matrix sizes.

    Args:
        model_type: Either 'transformer' or 'mamba'
        matrix_sizes: List of matrix dimensions (n_rows = n_cols)

    Returns:
        List of dictionaries containing benchmark results
    """
    results = []

    print(f"\nBenchmarking {model_type.upper()}...")
    for matrix_size in tqdm(matrix_sizes, desc=f"{model_type.upper()}"):
        n_rows = n_cols = matrix_size

        # Pre-generate channels for all batches (tokenization happens per batch during measurement)
        # Generate enough samples for all measurement runs
        total_samples = BATCH_SIZE * MEASURE_RUNS
        print(f"\n  Generating {matrix_size}x{matrix_size} matrices for {MEASURE_RUNS} batches...")
        channels = generate_synthetic_channels(total_samples, n_rows, n_cols)

        # Get sequence length from a sample tokenization
        sample_tokens = tokenize_channels(channels[:BATCH_SIZE])
        actual_seq_len = sample_tokens.shape[1]
        n_patches = actual_seq_len - 1  # Exclude CLS token
        del sample_tokens

        # Calculate max_len for model
        max_len = actual_seq_len + 10  # Add buffer

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

        # Measure ONLY model inference latency (not tokenization)
        try:
            latency = measure_latency(model, channels)
            results.append({
                "matrix_size": matrix_size,
                "sequence_length": actual_seq_len,
                "n_patches": n_patches,
                "mean_latency": latency["mean"],
                "std_latency": latency["std"]
            })

            print(f"  Matrix={matrix_size}x{matrix_size}, Seq_len={actual_seq_len}, "
                  f"Patches={n_patches}, Latency={latency['mean']:.2f}±{latency['std']:.2f}ms (batch_size={BATCH_SIZE})")

        except RuntimeError as e:
            print(f"\nFailed at matrix_size={matrix_size}: {str(e)}")

        # Cleanup
        del model, channels
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return results


def plot_results(transformer_results: List[Dict], mamba_results: List[Dict]):
    """Create publication-quality plot comparing latencies."""

    # Extract data and convert from ms to seconds
    trans_sizes = [r["matrix_size"] for r in transformer_results]
    trans_mean = [r["mean_latency"] / 1000.0 for r in transformer_results]  # Convert to seconds

    mamba_sizes = [r["matrix_size"] for r in mamba_results]
    mamba_mean = [r["mean_latency"] / 1000.0 for r in mamba_results]  # Convert to seconds

    # Set seaborn style
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.5)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot smooth curves
    ax.plot(trans_sizes, trans_mean,
            marker='o', markersize=8, linewidth=3,
            label='Transformer', color='#3498db',
            linestyle='-', markeredgewidth=0, alpha=0.85)

    ax.plot(mamba_sizes, mamba_mean,
            marker='s', markersize=8, linewidth=3,
            label='Mamba', color='#e74c3c',
            linestyle='-', markeredgewidth=0, alpha=0.85)

    # Styling
    ax.set_xlabel('Channel Matrix Size (N × N)', fontsize=18, fontweight='600')
    ax.set_ylabel('Inference Latency (s)', fontsize=18, fontweight='600')
    ax.set_title('Inference Latency',
                 fontsize=20, fontweight='700', pad=20)

    # Set x-ticks to exact matrix sizes
    ax.set_xticks(trans_sizes)
    ax.set_xticklabels([f'{size}' for size in trans_sizes])

    # Legend
    ax.legend(fontsize=16, frameon=True, fancybox=True,
              shadow=True, loc='upper left', framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)

    # Tick styling
    ax.tick_params(labelsize=14)

    # Remove top and right spines
    sns.despine()

    plt.tight_layout()

    # Save with high DPI
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs/plots/latency_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nPlot saved to: {output_file}")

    plt.show()


def main():
    """Main execution."""
    set_seed(SEED)

    print("=" * 80)
    print("LATENCY BENCHMARK: Mamba vs Transformer")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE} samples per batch")
    print(f"Measurement: Average over {MEASURE_RUNS} batches")
    print(f"Channel Matrix Sizes: {MATRIX_SIZES}")
    print(f"Model: d_model={D_MODEL}, n_layers={N_LAYERS}")
    print(f"Patch Size: {PATCH_ROWS}x{PATCH_COLS}")
    print(f"Note: Measuring ONLY model inference latency (tokenization excluded)")
    print("=" * 80)

    # Run benchmarks
    transformer_results = benchmark_model("transformer", MATRIX_SIZES)
    mamba_results = benchmark_model("mamba", MATRIX_SIZES)

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print("\nTransformer:")
    for r in transformer_results:
        print(f"  matrix={r['matrix_size']:3d}x{r['matrix_size']:3d} "
              f"(seq_len={r['sequence_length']:4d}, patches={r['n_patches']:4d}): "
              f"{r['mean_latency']:7.2f} ± {r['std_latency']:5.2f} ms")

    print("\nMamba:")
    for r in mamba_results:
        print(f"  matrix={r['matrix_size']:3d}x{r['matrix_size']:3d} "
              f"(seq_len={r['sequence_length']:4d}, patches={r['n_patches']:4d}): "
              f"{r['mean_latency']:7.2f} ± {r['std_latency']:5.2f} ms")

    print("\nSpeedup (Transformer / Mamba):")
    for i in range(min(len(transformer_results), len(mamba_results))):
        speedup = transformer_results[i]['mean_latency'] / mamba_results[i]['mean_latency']
        matrix_size = transformer_results[i]['matrix_size']
        print(f"  matrix={matrix_size:3d}x{matrix_size:3d}: {speedup:.2f}x")

    # Create plot
    print("\n" + "=" * 80)
    print("Generating plot...")
    plot_results(transformer_results, mamba_results)

    # Save results to CSV
    import csv
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs/plots/latency_benchmark.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['matrix_size', 'seq_length', 'n_patches',
                         'transformer_latency_ms', 'transformer_std_ms',
                         'mamba_latency_ms', 'mamba_std_ms', 'speedup'])
        for i in range(min(len(transformer_results), len(mamba_results))):
            tr = transformer_results[i]
            mr = mamba_results[i]
            speedup = tr['mean_latency'] / mr['mean_latency']
            writer.writerow([
                tr['matrix_size'], tr['sequence_length'], tr['n_patches'],
                f"{tr['mean_latency']:.2f}", f"{tr['std_latency']:.2f}",
                f"{mr['mean_latency']:.2f}", f"{mr['std_latency']:.2f}",
                f"{speedup:.2f}"
            ])
    print(f"\nCSV saved to: {csv_path}")

    print("=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
