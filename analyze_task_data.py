"""
Analyze Task Data

This script analyzes the test data in each task directory and reports
the unique channel sizes present in each task.
"""

import os
import torch
import numpy as np
from collections import defaultdict


def analyze_task_channels(task_dir: str) -> dict:
    """
    Analyze channel data in a task directory.

    Args:
        task_dir: Path to the task directory (e.g., 'task_1')

    Returns:
        Dictionary with channel size statistics
    """
    test_data_path = os.path.join(task_dir, "test_data.pt")

    if not os.path.exists(test_data_path):
        return {
            "exists": False,
            "error": f"File not found: {test_data_path}"
        }

    try:
        # Load test data
        test_data = torch.load(test_data_path, map_location='cpu')

        # Get channels
        channels = test_data.get("channels")
        if channels is None:
            return {
                "exists": True,
                "error": "No 'channels' key in test data"
            }

        # Analyze channel sizes
        # Channels shape should be (n_samples, n_rows, n_cols) for complex data
        # or (n_samples, n_channels, n_rows, n_cols) for real data

        shape = channels.shape
        n_samples = shape[0]

        # Determine channel dimensions
        if len(shape) == 3:
            # Complex format: (n_samples, n_rows, n_cols)
            unique_sizes = set()
            for i in range(n_samples):
                channel = channels[i]
                size = (channel.shape[0], channel.shape[1])
                unique_sizes.add(size)
        elif len(shape) == 4:
            # Real format: (n_samples, n_channels, n_rows, n_cols)
            unique_sizes = set()
            for i in range(n_samples):
                channel = channels[i]
                size = (channel.shape[1], channel.shape[2])
                unique_sizes.add(size)
        else:
            return {
                "exists": True,
                "error": f"Unexpected channel shape: {shape}"
            }

        # Convert to sorted list
        unique_sizes = sorted(list(unique_sizes))

        # Extract unique dimensions (assuming square or consistent sizes)
        unique_dims = sorted(list(set([size[0] for size in unique_sizes] + [size[1] for size in unique_sizes])))

        return {
            "exists": True,
            "n_samples": n_samples,
            "channel_shape": shape,
            "unique_sizes": unique_sizes,
            "unique_dimensions": unique_dims,
            "is_complex": channels.is_complex() if hasattr(channels, 'is_complex') else False
        }

    except Exception as e:
        return {
            "exists": True,
            "error": f"Error loading data: {str(e)}"
        }


def main():
    """Main execution."""

    print("=" * 80)
    print("TASK DATA ANALYSIS: Channel Sizes")
    print("=" * 80)
    print()

    # Analyze each task
    results = {}
    for task_num in range(1, 6):
        task_dir = f"task_{task_num}"
        print(f"Analyzing {task_dir}...", end=" ")

        result = analyze_task_channels(task_dir)
        results[task_dir] = result

        if not result["exists"]:
            print(f"❌ {result['error']}")
        elif "error" in result:
            print(f"❌ {result['error']}")
        else:
            print("✓")

    # Print summary
    print()
    print("=" * 80)
    print("CHANNEL SIZE SUMMARY")
    print("=" * 80)
    print()

    for task_dir in sorted(results.keys()):
        result = results[task_dir]

        if not result["exists"] or "error" in result:
            print(f"{task_dir}: Error - {result.get('error', 'Unknown error')}")
        else:
            dims = result["unique_dimensions"]
            dims_str = str(dims).replace("[", "").replace("]", "")
            print(f"{task_dir}: channel_sizes [{dims_str}]")

    # Print detailed information
    print()
    print("=" * 80)
    print("DETAILED INFORMATION")
    print("=" * 80)
    print()

    for task_dir in sorted(results.keys()):
        result = results[task_dir]

        if result["exists"] and "error" not in result:
            print(f"{task_dir}:")
            print(f"  Number of samples:    {result['n_samples']}")
            print(f"  Channel shape:        {result['channel_shape']}")
            print(f"  Is complex:           {result['is_complex']}")
            print(f"  Unique dimensions:    {result['unique_dimensions']}")

            if len(result['unique_sizes']) <= 5:
                print(f"  Unique (rows, cols):  {result['unique_sizes']}")
            else:
                print(f"  Number of unique sizes: {len(result['unique_sizes'])}")
                print(f"  Sample sizes:         {result['unique_sizes'][:5]} ...")
            print()

    print("=" * 80)


if __name__ == "__main__":
    main()
