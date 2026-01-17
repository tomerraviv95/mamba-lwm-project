"""
Compare Submission Results

This script reads score.json files from different model submissions and creates
a CSV comparing their performance across all tasks.
"""

import os
import json
import pandas as pd
from pathlib import Path


def read_score(submission_dir: str, task_num: int) -> float:
    """
    Read score from a specific task in a submission directory.

    Args:
        submission_dir: Path to the submission directory
        task_num: Task number (1-5)

    Returns:
        Score as float, or None if file not found
    """
    score_path = os.path.join(submission_dir, f"task_{task_num}", "score.json")

    if not os.path.exists(score_path):
        print(f"Warning: {score_path} not found")
        return None

    try:
        with open(score_path, 'r') as f:
            score = json.load(f)
        return float(score)
    except Exception as e:
        print(f"Error reading {score_path}: {e}")
        return None


def read_composite_score(submission_dir: str) -> float:
    """
    Read composite score from submission directory.

    Args:
        submission_dir: Path to the submission directory

    Returns:
        Composite score as float, or None if file not found
    """
    composite_path = os.path.join(submission_dir, "composite_score.json")

    if not os.path.exists(composite_path):
        print(f"Warning: {composite_path} not found")
        return None

    try:
        with open(composite_path, 'r') as f:
            score = json.load(f)
        return float(score)
    except Exception as e:
        print(f"Error reading {composite_path}: {e}")
        return None


def create_comparison_csv(
    submissions: dict,
    output_file: str = "model_comparison.csv"
):
    """
    Create a CSV comparing multiple submissions across all tasks.

    Args:
        submissions: Dictionary mapping model names to submission directories
        output_file: Output CSV filename
    """
    # Task names
    task_names = [
        "Task 1: LoS/NLoS Classification",
        "Task 2: Beam Prediction",
        "Task 3: Channel Interpolation",
        "Task 4: Channel Estimation",
        "Task 5: User Localization",
        "Composite Score"
    ]

    # Baseline scores
    baseline_scores = [0.9396, 0.6137, 0.4165, 0.4576, 0.6711]
    baseline_composite = sum(baseline_scores) / len(baseline_scores)

    # Prepare data - start with baseline
    data = []

    # Add baseline row
    baseline_row = {"Model": "Baseline"}
    for i, task_name in enumerate(task_names[:5]):
        baseline_row[task_name] = baseline_scores[i]
    baseline_row[task_names[5]] = baseline_composite
    data.append(baseline_row)

    # Add submission rows
    for model_name, submission_dir in submissions.items():
        row = {"Model": model_name}

        # Read scores for tasks 1-5
        for task_num in range(1, 6):
            score = read_score(submission_dir, task_num)
            task_name = task_names[task_num - 1]
            row[task_name] = score if score is not None else "N/A"

        # Read composite score
        composite = read_composite_score(submission_dir)
        row[task_names[5]] = composite if composite is not None else "N/A"

        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Set Model as index
    df.set_index("Model", inplace=True)

    # Round all numeric values to 3 decimal places
    for col in df.columns:
        df[col] = df[col].apply(lambda x: round(x, 3) if isinstance(x, (int, float)) else x)

    # Save to CSV with 3 decimal places
    df.to_csv(output_file, float_format='%.3f')
    print(f"\nComparison table saved to: {output_file}")

    # Print to console with 3 decimal places
    print("\n" + "=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)
    print(df.to_string(float_format=lambda x: f"{x:.3f}"))
    print("=" * 100)

    return df


def main():
    """Main execution."""

    # Define submissions to compare
    submissions = {
        "Transformer (epoch55)": "submission_transformer_lwm_epoch55_train42238",
        "Mamba (epoch13)": "submission_mamba_lwm_epoch13_train32124"
    }

    print("=" * 100)
    print("SUBMISSION COMPARISON")
    print("=" * 100)
    print(f"\nComparing {len(submissions)} submissions:")
    for model_name, submission_dir in submissions.items():
        print(f"  - {model_name}: {submission_dir}")
    print()

    # Check if directories exist
    for model_name, submission_dir in submissions.items():
        if not os.path.exists(submission_dir):
            print(f"ERROR: Directory not found: {submission_dir}")
            return

    # Create comparison CSV
    df = create_comparison_csv(submissions, output_file="model_comparison.csv")

    # Calculate and display differences
    if len(submissions) == 2:
        print("\n" + "=" * 100)
        print("PERFORMANCE DIFFERENCE (Mamba - Transformer)")
        print("=" * 100)

        model_names = list(submissions.keys())
        differences = {}

        for col in df.columns:
            val1 = df.loc[model_names[0], col]
            val2 = df.loc[model_names[1], col]

            if val1 != "N/A" and val2 != "N/A":
                diff = val1 - val2
                differences[col] = diff
                sign = "+" if diff >= 0 else ""
                print(f"  {col:40s}: {sign}{diff:.3f}")
            else:
                print(f"  {col:40s}: N/A")

        print("=" * 100)


if __name__ == "__main__":
    main()
