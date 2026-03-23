"""
Wrapper script to run sample variation experiments across multiple patch sizes.
Runs train_heads.py with different PATCH_SIZE values and organizes results.
"""
import os
import subprocess
import shutil
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_SCRIPT_DIR, '..')

# Configuration
PATCH_SIZES = [4, 6, 8]
MODEL_TYPE = "raw"  # Change this to "transformer" or "raw" as needed

print(f"\n{'='*70}")
print(f" Running Sample Variation Experiments")
print(f" Model: {MODEL_TYPE}")
print(f" Patch Sizes: {PATCH_SIZES}")
print(f"{'='*70}\n")

total_start = time.time()

for patch_size in PATCH_SIZES:
    print(f"\n{'#'*70}")
    print(f"# PATCH SIZE: {patch_size}x{patch_size}")
    print(f"{'#'*70}\n")

    patch_start = time.time()

    # Read train_heads.py
    train_heads_path = os.path.join(_SCRIPT_DIR, 'train_heads.py')
    with open(train_heads_path, 'r') as f:
        content = f.read()

    # Update MODEL_TYPE (use regex to handle any current MODEL_TYPE value)
    import re as _re
    content = _re.sub(
        r'MODEL_TYPE = "[^"]*"  # Options:.*',
        f'MODEL_TYPE = "{MODEL_TYPE}"  # Options: "transformer", "mamba", "raw", "wimae", "contrawimae"',
        content
    )

    # Update PATCH_SIZE
    import re
    content = re.sub(
        r'PATCH_SIZE = \d+',
        f'PATCH_SIZE = {patch_size}',
        content
    )

    # Write modified file
    with open(train_heads_path, 'w') as f:
        f.write(content)

    # Run training
    print(f"Running training with patch size {patch_size}x{patch_size}...")
    result = subprocess.run(['python', train_heads_path], capture_output=False)

    if result.returncode != 0:
        print(f"ERROR: Training failed for patch size {patch_size}")
        continue

    # Move results to patch-specific directory
    source_dir = os.path.join(_REPO_ROOT, f"outputs/submissions/submission_sample_variation_{MODEL_TYPE}")
    target_dir = os.path.join(_REPO_ROOT, f"outputs/submissions/submission_sample_variation_{MODEL_TYPE}_patch{patch_size}")

    if os.path.exists(source_dir):
        # Remove target if it exists
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)

        # Move results
        shutil.move(source_dir, target_dir)
        print(f"\nMoved results to: {target_dir}")

    patch_time = time.time() - patch_start
    print(f"\nPatch size {patch_size}x{patch_size} completed in {patch_time/60:.1f} minutes")

total_time = time.time() - total_start

print(f"\n{'='*70}")
print(f" All experiments completed!")
print(f" Total time: {total_time/60:.1f} minutes")
print(f"{'='*70}")

print(f"\nResults saved in:")
for patch_size in PATCH_SIZES:
    target_dir = f"submission_sample_variation_{MODEL_TYPE}_patch{patch_size}"
    if os.path.exists(target_dir):
        print(f"  - {target_dir}/")

print(f"\nTo generate plots, run:")
print(f"  python plot_sample_variation.py")
