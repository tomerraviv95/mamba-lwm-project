# =============================================================================
# CONTINUED PRETRAINING SCRIPT: ADAPT TO DIFFERENT PATCH SIZES
#
# This script allows continued pretraining of an existing LWM model (transformer or mamba)
# with a different patch size than the original training.
#
# Usage:
#   1. Set MODEL_ARCHITECTURE ("transformer" or "mamba")
#   2. Set CHECKPOINT_PATH to your pretrained model
#   3. Set TARGET_PATCH_SIZE (e.g., 8, 16, 32, 64)
#   4. Run: python continue_pretrain.py
#
# The script will:
#   - Load the pretrained model weights (except embedding layer)
#   - Initialize a new embedding layer for the target patch size
#   - Run continued pretraining with new tokenization
#   - Save the adapted model
# =============================================================================

# Fix matplotlib backend to avoid tkinter errors on Windows
import matplotlib
matplotlib.use("Agg")

import os
import pickle
import random
import warnings
from collections import defaultdict

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

import pretrained_model
import mamba_model
from utils import (
    count_parameters,
    create_train_dataloader,
    generate_channels_and_labels,
    patch_maker,
    make_sample,
    train_lwm,
)

warnings.filterwarnings("ignore", category=UserWarning)

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model selection
MODEL_ARCHITECTURE = "transformer"  # or "mamba"
# CHECKPOINT_PATH = os.path.join(_REPO_ROOT, "outputs/submissions/submission_mamba_lwm_epoch13_train32124/model_checkpoint.pth")
CHECKPOINT_PATH = os.path.join(_REPO_ROOT, "outputs/submissions/submission_transformer_model_checkpoint/model_checkpoint.pth")

# Patch size configuration
SOURCE_PATCH_SIZE = 4  # Original patch size model was trained with (4x4)
TARGET_PATCH_SIZE = 8  # New patch size to adapt to (e.g., 8, 16, 32, 64)

# Calculated element lengths
SOURCE_ELEMENT_LENGTH = SOURCE_PATCH_SIZE * SOURCE_PATCH_SIZE * 2  # 32 for 4x4
TARGET_ELEMENT_LENGTH = TARGET_PATCH_SIZE * TARGET_PATCH_SIZE * 2  # e.g., 128 for 8x8

# Model hyperparameters (must match checkpoint)
D_MODEL = 128
N_LAYERS = 12
MAX_LEN = 513
N_HEADS = 8  # For transformer
DROPOUT = 0.1

# Mamba-specific (if using mamba)
D_STATE = 8
D_CONV = 4
EXPAND = 1.2
BIDIRECTIONAL = True

# Continued pretraining settings
CONTINUE_EPOCHS = 10  # Short continued pretraining phase
BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
WARMUP_EPOCHS = 0
BASE_LR = 5e-5  # Lower learning rate for continued pretraining
MIN_LR = 1e-8
MASK_PERCENT = 0.4
WEIGHT_DECAY = 0.05
BETA1 = 0.9
BETA2 = 0.999

# Validation schedule
VALIDATION_EPOCHS = [1,3,5,8,10]  # Only run validation on these specific epochs

# Save directory
SAVE_DIR = os.path.join(_REPO_ROOT, f"outputs/pretrained_models/pretrained_models_{MODEL_ARCHITECTURE}_patch{TARGET_PATCH_SIZE}x{TARGET_PATCH_SIZE}")

# Filter for specific sequence lengths (set to None or [] to use all)
FILTER_SEQ_LENGTHS = [17, 33, 65, 129, 257]   # Set to [17, 33, 65, 129, 257] to filter, or None to use all

# =============================================================================
# CUSTOM TOKENIZATION FUNCTION WITH CONFIGURABLE PATCH SIZE
# =============================================================================

def tokenizer_train_custom(channels,
                           max_len=513,
                           masking_percent=0.40,
                           mask=False,
                           seed=42,
                           patch_rows=4,
                           patch_cols=4):
    """
    Custom tokenizer that accepts patch_rows and patch_cols parameters.
    Based on utils.tokenizer_train but with configurable patch size.

    Args:
        channels (list): List of channel arrays to tokenize
        max_len (int): Maximum sequence length
        masking_percent (float): Percentage of patches to mask
        mask (bool): Whether to apply masking
        seed (int): Random seed
        patch_rows (int): Number of rows per patch
        patch_cols (int): Number of columns per patch

    Returns:
        dict or torch.Tensor: Tokenized samples grouped by sequence length (if mask=True)
                             or stacked tensor (if mask=False)
    """
    # Generate patches with custom patch size
    patches = [patch_maker(channel_set, patch_rows=patch_rows, patch_cols=patch_cols)
               for channel_set in channels]
    patches = [patch for patch_list in patches for patch in patch_list]
    print(f"\nTotal number of samples: {len(patches)}")
    print(f"Using patch size: {patch_rows}x{patch_cols}")

    grouped_data = defaultdict(list)  # Group samples by sequence length
    grouped_data_2 = []

    for user_idx in tqdm(range(len(patches)), desc="Processing items"):
        patch_size = patches[user_idx].shape[1]
        n_patches = patches[user_idx].shape[0]
        n_masks_half = int(masking_percent * n_patches)

        # Word2id dictionary with special tokens sized for the patch dimension
        word2id = {
            '[CLS]': 0.2 * np.ones((patch_size)),
            '[MASK]': 0.1 * np.ones((patch_size))
        }

        sample = make_sample(
            user_idx, patches, word2id, n_patches, n_masks_half, patch_size, mask=mask, seed=seed
        )

        if mask:
            seq_length = len(sample[0])
            grouped_data[seq_length].append(sample)
        else:
            grouped_data_2.append(sample)

    if mask:
        normalized_grouped_data = {key: grouped_data[key] for key in sorted(grouped_data.keys())}
    else:
        normalized_grouped_data = torch.stack(grouped_data_2, dim=0)

    return normalized_grouped_data


def save_tokenized_to_hdf5(hdf5_path, tokenized_data_list, seq_length, chunk_size=32):
    """
    Save tokenized data to HDF5 file with optimal chunking for random access.

    Args:
        hdf5_path (str): Path to HDF5 file
        tokenized_data_list (list): List of [input_ids, masked_tokens, masked_pos] samples
        seq_length (int): Sequence length for this group
        chunk_size (int): Number of samples per chunk (default: 32)
    """
    num_samples = len(tokenized_data_list)
    if num_samples == 0:
        return

    # Convert first sample to numpy to get shapes
    sample = tokenized_data_list[0]
    input_ids_np = np.array(sample[0], dtype=np.float32)
    masked_tokens_np = np.array(sample[1], dtype=np.float32)
    masked_pos_np = np.array(sample[2], dtype=np.int32)

    input_ids_shape = input_ids_np.shape
    masked_tokens_shape = masked_tokens_np.shape
    masked_pos_shape = masked_pos_np.shape

    # Prepare arrays - convert all samples to numpy
    input_ids_array = np.array([np.array(s[0], dtype=np.float32) for s in tokenized_data_list], dtype=np.float32)
    masked_tokens_array = np.array([np.array(s[1], dtype=np.float32) for s in tokenized_data_list], dtype=np.float32)
    masked_pos_array = np.array([np.array(s[2], dtype=np.int32) for s in tokenized_data_list], dtype=np.int32)

    # Create or append to HDF5 file
    with h5py.File(hdf5_path, 'a') as f:
        # Create group for this sequence length
        group_name = f'length_{seq_length}'

        if group_name in f:
            # Group already exists, remove it to overwrite
            del f[group_name]

        group = f.create_group(group_name)

        # Create datasets with chunking and compression
        group.create_dataset(
            'input_ids',
            data=input_ids_array,
            chunks=(min(chunk_size, num_samples), *input_ids_shape),
            compression='gzip',
            compression_opts=4,
            dtype=np.float32
        )

        group.create_dataset(
            'masked_tokens',
            data=masked_tokens_array,
            chunks=(min(chunk_size, num_samples), *masked_tokens_shape),
            compression='gzip',
            compression_opts=4,
            dtype=np.float32
        )

        group.create_dataset(
            'masked_pos',
            data=masked_pos_array,
            chunks=(min(chunk_size, num_samples), *masked_pos_shape),
            compression='gzip',
            compression_opts=4,
            dtype=np.int32
        )

        # Store metadata
        group.attrs['num_samples'] = num_samples
        group.attrs['seq_length'] = seq_length
        group.attrs['input_ids_shape'] = input_ids_shape
        group.attrs['masked_tokens_shape'] = masked_tokens_shape
        group.attrs['masked_pos_shape'] = masked_pos_shape


# =============================================================================
# CUSTOM TRAINING FUNCTION WITH CUSTOM VALIDATION SCHEDULE
# =============================================================================

def train_lwm_custom_validation(model, train_loaders, val_loaders, optimizer, scheduler, epochs,
                                validation_epochs, device, save_dir="models",
                                log_file="training_log.csv", max_batches_per_epoch=None):
    """
    Custom training function that validates only on specific epochs.

    Based on utils.train_lwm but with custom validation schedule.

    Args:
        model: Model to train
        train_loaders: Dictionary of training dataloaders
        val_loaders: Dictionary of validation dataloaders
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epochs: Total number of epochs
        validation_epochs: List of epochs to run validation (e.g., [5, 10])
        device: Device to train on
        save_dir: Directory to save checkpoints
        log_file: CSV log file name
        max_batches_per_epoch: Maximum batches per epoch per loader
    """
    import csv
    import matplotlib.pyplot as plt
    from utils import nmse_loss

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Initialize loss criterion
    criterion = nn.MSELoss(reduction='sum')

    # Initialize lists to store losses
    train_mse_losses = []
    val_mse_losses = []
    val_nmse_losses = []
    best_val_mse = float('inf')

    # Initialize CSV log file
    log_path = os.path.join(save_dir, log_file)
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_mse', 'val_mse', 'val_nmse', 'learning_rate'])

    for epoch in range(epochs):
        # Training loop
        model.train()
        train_mse = 0.0
        train_samples = 0

        print(f"\nEpoch {epoch + 1}/{epochs} [Training]")
        if max_batches_per_epoch:
            print(f"  Max batches per dataloader: {max_batches_per_epoch}")

        for length, train_loader in train_loaders.items():
            print(f"Processing sequences of length {length}")

            total_batches = len(train_loader) if not max_batches_per_epoch else min(len(train_loader), max_batches_per_epoch)

            with tqdm(train_loader, desc=f"Length {length} [Training]", unit="batch", total=total_batches) as t:
                for batch_idx, batch in enumerate(t):
                    if max_batches_per_epoch and batch_idx >= max_batches_per_epoch:
                        print(f"  Reached max_batches_per_epoch ({max_batches_per_epoch}) for length {length}")
                        break

                    optimizer.zero_grad()

                    # Move data to device
                    input_ids, masked_tokens, masked_pos = [b.to(device) for b in batch]

                    # Forward pass
                    logits_lm = model(input_ids, masked_pos)[0]

                    # Compute MSE loss
                    loss = criterion(masked_tokens, logits_lm)

                    # Check for NaN/Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"\n{'='*80}")
                        print(f"❌ NaN/Inf DETECTED IN LOSS at Epoch {epoch}, Length {length}")
                        print(f"{'='*80}")
                        print(f"Input stats - Min: {input_ids.min().item():.4f}, Max: {input_ids.max().item():.4f}, Mean: {input_ids.mean().item():.4f}")
                        print(f"Logits stats - Min: {logits_lm.min().item():.4f}, Max: {logits_lm.max().item():.4f}, Mean: {logits_lm.mean().item():.4f}")
                        print(f"{'='*80}\n")
                        exit()

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    train_mse += loss.item()
                    train_samples += input_ids.shape[0]

                    # Update progress bar
                    t.set_postfix({"mse": train_mse / train_samples, "lr": scheduler.get_last_lr()[0]})

        # Average MSE across training samples
        train_mse = train_mse / max(train_samples, 1)
        train_mse_losses.append(train_mse)

        # Validation loop - only on specified epochs
        val_mse = None
        val_nmse = None

        if (epoch + 1) in validation_epochs:  # epoch is 0-indexed, validation_epochs is 1-indexed
            model.eval()
            val_mse = 0.0
            val_nmse = 0.0
            val_samples = 0

            with torch.no_grad():
                print(f"\nEpoch {epoch + 1}/{epochs} [Validation]")
                for length, val_loader in val_loaders.items():
                    print(f"Processing sequences of length {length}")

                    # Limit validation batches same as training
                    total_batches = len(val_loader) if not max_batches_per_epoch else min(len(val_loader), max_batches_per_epoch)

                    with tqdm(val_loader, desc=f"Length {length} [Validation]", unit="batch", total=total_batches) as t:
                        for batch_idx, batch in enumerate(t):
                            # Stop after max_batches_per_epoch if limit is set
                            if max_batches_per_epoch and batch_idx >= max_batches_per_epoch:
                                print(f"  Reached max_batches_per_epoch ({max_batches_per_epoch}) for length {length}")
                                break

                            # Move data to device
                            input_ids, masked_tokens, masked_pos = [b.to(device) for b in batch]

                            # Forward pass
                            logits_lm = model(input_ids, masked_pos)[0]

                            # Compute MSE loss
                            mse = criterion(masked_tokens, logits_lm)
                            val_mse += mse.item()

                            # Compute NMSE
                            masked_tokens_np = masked_tokens.cpu().numpy()
                            logits_lm_np = logits_lm.cpu().numpy()
                            nmse = nmse_loss(masked_tokens_np, logits_lm_np)
                            val_nmse += nmse * input_ids.shape[0]

                            val_samples += input_ids.shape[0]

                            # Update progress bar
                            t.set_postfix({"mse": val_mse / val_samples, "nmse": val_nmse / val_samples})

            # Average MSE and NMSE
            val_mse = val_mse / max(val_samples, 1)
            val_nmse = val_nmse / max(val_samples, 1)
            val_mse_losses.append(val_mse)
            val_nmse_losses.append(val_nmse)

            # Save model if validation MSE improves
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                model_path = os.path.join(save_dir, f"lwm_epoch{epoch+1}_train{train_mse:.4f}_val{val_mse:.4f}.pth")
                torch.save(model.state_dict(), model_path)
                print(f"✓ Model saved: {model_path}")

        # Log results to console
        print(f"  Train MSE: {train_mse:.4f}")
        if val_mse is not None:
            print(f"  Validation MSE: {val_mse:.4f}")
            print(f"  Validation NMSE: {val_nmse:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6e}")

        # Log to CSV file
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if val_mse is not None:
                writer.writerow([epoch + 1, train_mse, val_mse, val_nmse, scheduler.get_last_lr()[0]])
            else:
                writer.writerow([epoch + 1, train_mse, '', '', scheduler.get_last_lr()[0]])

        # Plot losses
        if train_mse_losses:
            plt.figure(figsize=(10, 6))
            epochs_train = list(range(1, len(train_mse_losses) + 1))
            plt.plot(epochs_train, train_mse_losses, label="Train MSE", linewidth=2)

            if val_mse_losses:
                # Plot validation at the epochs where it was computed
                val_epochs = validation_epochs[:len(val_mse_losses)]
                plt.plot(val_epochs, val_mse_losses, label="Validation MSE", marker='o', linewidth=2, markersize=6)
                plt.plot(val_epochs, val_nmse_losses, label="Validation NMSE", marker='s', linewidth=2, markersize=6)

            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Training and Validation Losses")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plot_path = os.path.join(save_dir, "training_progress.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"Best Validation MSE: {best_val_mse:.4f}")
    print(f"{'='*80}")

    return model


# =============================================================================
# SCENARIO DEFINITIONS (Same as train_lwm.py)
# =============================================================================

def scenarios_list():
    scen_list = np.array(
        [
            "city_0_newyork_3p5_lwm",
            "city_1_losangeles_3p5_lwm",
            "city_2_chicago_3p5_lwm",
            "city_3_houston_3p5_lwm",
            "city_4_phoenix_3p5_lwm",
            "city_5_philadelphia_3p5_lwm",
            "city_6_miami_3p5_lwm",
            "city_7_sandiego_3p5_lwm",
            "city_8_dallas_3p5_lwm",
            "city_9_sanfrancisco_3p5_lwm",
            "city_10_austin_3p5_lwm",
            "city_11_santaclara_3p5_lwm",
            "city_12_fortworth_3p5_lwm",
            "city_13_columbus_3p5_lwm",
            "city_14_charlotte_3p5_lwm",
            "city_15_indianapolis_3p5_lwm",
            "city_16_sanfrancisco_3p5_lwm",
            "city_17_seattle_3p5_lwm",
            "city_18_denver_3p5_lwm",
            "city_19_oklahoma_3p5_lwm",
            "asu_campus_3p5",
            "o1_3p5",
            "boston5G_3p5",
        ]
    )
    return scen_list


def scenario_prop():
    # Import the full scenario properties from train_lwm.py
    # (Abbreviated here for brevity - use the same dict as in train_lwm.py)
    from train_lwm import scenario_prop as get_scenario_prop
    return get_scenario_prop()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CONTINUED PRETRAINING WITH DIFFERENT PATCH SIZE")
    print("=" * 80)
    print(f"Model Architecture: {MODEL_ARCHITECTURE}")
    print(f"Source Checkpoint: {CHECKPOINT_PATH}")
    print(f"Source Patch Size: {SOURCE_PATCH_SIZE}x{SOURCE_PATCH_SIZE} (element_length={SOURCE_ELEMENT_LENGTH})")
    print(f"Target Patch Size: {TARGET_PATCH_SIZE}x{TARGET_PATCH_SIZE} (element_length={TARGET_ELEMENT_LENGTH})")
    print(f"Continued Pretraining Epochs: {CONTINUE_EPOCHS}")
    print(f"Learning Rate: {BASE_LR}")
    print(f"Save Directory: {SAVE_DIR}")
    print("=" * 80)

    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Set seed
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Device setup
    gpu_ids = [0]
    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # =============================================================================
    # 1. INITIALIZE MODEL WITH NEW ELEMENT LENGTH
    # =============================================================================

    print("\n" + "=" * 80)
    print("1. INITIALIZING MODEL WITH NEW PATCH SIZE")
    print("=" * 80)

    if MODEL_ARCHITECTURE.lower() == "transformer":
        model = pretrained_model.lwm(
            element_length=TARGET_ELEMENT_LENGTH,
            d_model=D_MODEL,
            n_layers=N_LAYERS,
            max_len=MAX_LEN,
            n_heads=N_HEADS,
            dropout=DROPOUT,
        ).to(device)
    elif MODEL_ARCHITECTURE.lower() == "mamba":
        model = mamba_model.lwm_mamba(
            element_length=TARGET_ELEMENT_LENGTH,
            d_model=D_MODEL,
            n_layers=N_LAYERS,
            max_len=MAX_LEN,
            d_state=D_STATE,
            d_conv=D_CONV,
            expand=EXPAND,
            dropout=DROPOUT,
            bidirectional=BIDIRECTIONAL,
        ).to(device)
    else:
        raise ValueError(f"Unknown model architecture: {MODEL_ARCHITECTURE}")

    print(f"Model initialized with element_length={TARGET_ELEMENT_LENGTH}")
    n_parameters = count_parameters(model)
    print(f"Number of trainable parameters: {n_parameters:,}")

    # =============================================================================
    # 2. LOAD PRETRAINED WEIGHTS (PARTIAL LOADING)
    # =============================================================================

    print("\n" + "=" * 80)
    print("2. LOADING PRETRAINED WEIGHTS (EXCLUDING EMBEDDING LAYER)")
    print("=" * 80)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Please verify the checkpoint path and try again.")
        exit(1)

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    # Remove DataParallel "module." prefix if present
    clean_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}

    # Get model's current state dict
    model_state_dict = model.state_dict()

    # Filter out embedding layer weights (they have wrong dimensions)
    if MODEL_ARCHITECTURE.lower() == "transformer":
        # Exclude: embedding.proj.weight, embedding.proj.bias
        keys_to_exclude = ["embedding.proj.weight", "embedding.proj.bias"]
    else:  # mamba
        # Exclude: proj.weight, proj.bias (input projection layer)
        # Note: decoder projects from d_model back to element_length, so it also needs to change
        keys_to_exclude = ["proj.weight", "proj.bias", "decoder.weight", "decoder.bias"]

    # Load compatible weights
    compatible_state_dict = {}
    skipped_keys = []

    for key, value in clean_state_dict.items():
        if key in keys_to_exclude:
            skipped_keys.append(f"{key} (excluded for new patch size)")
            continue

        if key not in model_state_dict:
            skipped_keys.append(f"{key} (not in current model)")
            continue

        if model_state_dict[key].shape != value.shape:
            skipped_keys.append(f"{key} (shape mismatch: {value.shape} vs {model_state_dict[key].shape})")
            continue

        compatible_state_dict[key] = value

    # Load the compatible weights
    model.load_state_dict(compatible_state_dict, strict=False)

    print(f"✓ Loaded {len(compatible_state_dict)}/{len(clean_state_dict)} layers from checkpoint")
    print(f"✓ Skipped {len(skipped_keys)} layers:")
    for key in skipped_keys:
        print(f"  - {key}")
    print(f"✓ Embedding layer initialized randomly for new patch size {TARGET_PATCH_SIZE}x{TARGET_PATCH_SIZE}")

    # =============================================================================
    # FREEZE ALL LAYERS EXCEPT EMBEDDING
    # =============================================================================

    print("\n" + "=" * 80)
    print("FREEZING PRETRAINED LAYERS")
    print("=" * 80)

    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Then, unfreeze only the embedding layer parameters
    trainable_params = []
    total_params = 0

    if MODEL_ARCHITECTURE.lower() == "transformer":
        # For transformer: unfreeze embedding.proj and embedding.pos_embed
        for name, param in model.named_parameters():
            if 'embedding.proj' in name or 'embedding.pos_embed' in name:
                param.requires_grad = True
                trainable_params.append(name)
        print("✓ Unfrozen transformer embedding layers:")
        print("  - embedding.proj (linear projection)")
        print("  - embedding.pos_embed (positional encoding)")
    else:  # mamba
        # For mamba: unfreeze proj and decoder
        for name, param in model.named_parameters():
            if 'proj.weight' in name or 'proj.bias' in name or 'decoder.weight' in name or 'decoder.bias' in name:
                param.requires_grad = True
                trainable_params.append(name)
        print("✓ Unfrozen mamba layers:")
        print("  - proj (input projection)")
        print("  - decoder (output projection)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params_count = total_params - trainable_params_count

    print(f"\nParameter summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params_count:,} ({100*trainable_params_count/total_params:.2f}%)")
    print(f"  Frozen parameters: {frozen_params_count:,} ({100*frozen_params_count/total_params:.2f}%)")
    print("=" * 80)

    # Wrap with DataParallel
    model = nn.DataParallel(model, device_ids=gpu_ids)

    # =============================================================================
    # 3. LOAD AND TOKENIZE DATA WITH NEW PATCH SIZE
    # =============================================================================

    print("\n" + "=" * 80)
    print(f"3. LOADING AND TOKENIZING DATA (Patch Size: {TARGET_PATCH_SIZE}x{TARGET_PATCH_SIZE})")
    print("=" * 80)

    scenarios = scenarios_list()
    scenario_properties = scenario_prop()

    # Create cache directories
    cache_dir = "/mnt/c/Users/tomer/scenarios_cache_lwm"
    os.makedirs(cache_dir, exist_ok=True)

    tokenized_base_dir = f"/mnt/c/Users/tomer/tokenized_data_lwm_patch{TARGET_PATCH_SIZE}x{TARGET_PATCH_SIZE}"
    os.makedirs(tokenized_base_dir, exist_ok=True)

    print(f"Tokenized data will be saved to: {tokenized_base_dir}")

    # Process scenarios (abbreviated for continued pretraining - use fewer scenarios)
    task = None  # No task-specific processing

    for scenario in scenarios[:-3]:  # Process city scenarios
        for bs_idx in range(1, 4):
            # Create cache filename for raw channels
            cache_filename = f"{scenario}_bs{bs_idx}.pkl"
            cache_filepath = os.path.join(cache_dir, cache_filename)

            # Check if cached data exists
            if os.path.exists(cache_filepath):
                print(f"\nLoading cached data for scenario: {scenario}, BS #{bs_idx}")
                with open(cache_filepath, "rb") as f:
                    cached_data = pickle.load(f)
                    scenario_channels = cached_data["channels"]
                    scenario_labels = cached_data["labels"]
            else:
                # Generate data if cache doesn't exist
                print(f"\nGenerating data for scenario: {scenario}, BS #{bs_idx}")
                scenario_channels, scenario_labels = generate_channels_and_labels(
                    n_ant_bs=scenario_properties[scenario]["n_ant_bs"],
                    n_subcarriers=scenario_properties[scenario]["n_subcarriers"],
                    bs_idx=bs_idx,
                    scenario_name=scenario,
                    task=task,
                    n_beams=64,
                )

                # Save to cache
                print(f"Saving data to cache: {cache_filepath}")
                with open(cache_filepath, "wb") as f:
                    pickle.dump(
                        {"channels": scenario_channels, "labels": scenario_labels}, f, protocol=5
                    )

            # Check if scenario has any valid channels
            if len(scenario_channels) == 0:
                print(f"Skipping scenario: {scenario}, BS #{bs_idx} (no valid channels)")
                continue

            # Check if tokenized HDF5 file already exists for this patch size
            hdf5_filename = f"{scenario}_bs{bs_idx}.h5"
            hdf5_filepath = os.path.join(tokenized_base_dir, hdf5_filename)

            if os.path.exists(hdf5_filepath):
                print(f"Tokenized data already exists for {scenario}, BS #{bs_idx}, skipping tokenization")
                continue

            # Tokenize the data with custom patch size
            print(f"Tokenizing scenario: {scenario}, BS #{bs_idx} with patch size {TARGET_PATCH_SIZE}x{TARGET_PATCH_SIZE}")
            scenario_preprocessed_dict = tokenizer_train_custom(
                [scenario_channels],
                max_len=MAX_LEN,
                masking_percent=MASK_PERCENT,
                mask=True,
                seed=42,
                patch_rows=TARGET_PATCH_SIZE,
                patch_cols=TARGET_PATCH_SIZE,
            )

            # Save tokenized data to HDF5 file
            print(f"Saving tokenized data to: {hdf5_filepath}")
            for seq_length, tokenized_samples in scenario_preprocessed_dict.items():
                save_tokenized_to_hdf5(
                    hdf5_filepath,
                    tokenized_samples,
                    seq_length=int(seq_length),
                    chunk_size=32
                )
                print(f"  Saved {len(tokenized_samples)} samples for sequence length {seq_length}")

    # Process campus/boston scenarios with zones (abbreviated)
    bs_idxs = [[1], [4, 15], [2]]
    for scenario_idx, scenario in enumerate(scenarios[-3:]):
        for bs_idx in bs_idxs[scenario_idx]:
            for zone in range(20):
                row_start = scenario_properties[scenario + f"_v{zone + 1}"]["n_rows"][0]
                row_end = scenario_properties[scenario + f"_v{zone + 1}"]["n_rows"][1]
                grid_idx = scenario_properties[scenario + f"_v{zone + 1}"]["grid_idx"] - 1

                # Cache filename
                cache_filename = f"{scenario}_bs{bs_idx}_zone{zone}.pkl"
                cache_filepath = os.path.join(cache_dir, cache_filename)

                # Check if cached data exists
                if os.path.exists(cache_filepath):
                    print(f"\nLoading cached data for scenario: {scenario}, BS #{bs_idx}, Zone {zone}")
                    with open(cache_filepath, "rb") as f:
                        cached_data = pickle.load(f)
                        scenario_channels = cached_data["channels"]
                        scenario_labels = cached_data["labels"]
                else:
                    # Generate data if cache doesn't exist
                    print(f"\nGenerating data for scenario: {scenario}, BS #{bs_idx}, Zone {zone}")
                    scenario_channels, scenario_labels = generate_channels_and_labels(
                        n_ant_bs=scenario_properties[scenario + f"_v{zone + 1}"]["n_ant_bs"],
                        n_subcarriers=scenario_properties[scenario + f"_v{zone + 1}"]["n_subcarriers"],
                        grid_idx=grid_idx,
                        bs_idx=bs_idx,
                        scenario_name=scenario,
                        rows=np.arange(row_start, row_end),
                        task=task,
                        n_beams=64,
                    )

                    # Save to cache
                    print(f"Saving data to cache: {cache_filepath}")
                    with open(cache_filepath, "wb") as f:
                        pickle.dump(
                            {"channels": scenario_channels, "labels": scenario_labels},
                            f,
                            protocol=5
                        )

                # Check if scenario has any valid channels
                if len(scenario_channels) == 0:
                    print(f"Skipping scenario: {scenario}, BS #{bs_idx}, Zone {zone} (no valid channels)")
                    continue

                # Check if tokenized HDF5 file already exists
                hdf5_filename = f"{scenario}_bs{bs_idx}_zone{zone}.h5"
                hdf5_filepath = os.path.join(tokenized_base_dir, hdf5_filename)

                if os.path.exists(hdf5_filepath):
                    print(f"Tokenized data already exists for {scenario}, BS #{bs_idx}, Zone {zone}, skipping tokenization")
                    continue

                # Tokenize the data with custom patch size
                print(f"Tokenizing scenario: {scenario}, BS #{bs_idx}, Zone {zone} with patch size {TARGET_PATCH_SIZE}x{TARGET_PATCH_SIZE}")
                scenario_preprocessed_dict = tokenizer_train_custom(
                    [scenario_channels],
                    max_len=MAX_LEN,
                    masking_percent=MASK_PERCENT,
                    mask=True,
                    seed=42,
                    patch_rows=TARGET_PATCH_SIZE,
                    patch_cols=TARGET_PATCH_SIZE,
                )

                # Save tokenized data to HDF5 file
                print(f"Saving tokenized data to: {hdf5_filepath}")
                for seq_length, tokenized_samples in scenario_preprocessed_dict.items():
                    save_tokenized_to_hdf5(
                        hdf5_filepath,
                        tokenized_samples,
                        seq_length=int(seq_length),
                        chunk_size=32
                    )
                    print(f"  Saved {len(tokenized_samples)} samples for sequence length {seq_length}")

    # =============================================================================
    # 4. COLLECT TOKENIZED DATA FILE METADATA
    # =============================================================================

    print("\n" + "=" * 80)
    print("4. COLLECTING TOKENIZED DATA FILE METADATA")
    print("=" * 80)

    # Apply sequence length filter if specified
    if FILTER_SEQ_LENGTHS:
        print(f"\nFiltering to specific sequence lengths: {sorted(FILTER_SEQ_LENGTHS)}")
        filter_keys = set(str(length) for length in FILTER_SEQ_LENGTHS)
    else:
        print("\nUsing all sequence lengths")
        filter_keys = None

    file_metadata = defaultdict(list)  # {seq_len: [(filepath, group_name, num_samples), ...]}

    # Iterate through all HDF5 files in tokenized_base_dir
    for filename in os.listdir(tokenized_base_dir):
        if filename.endswith(".h5"):
            hdf5_filepath = os.path.join(tokenized_base_dir, filename)

            print(f"\nProcessing HDF5 file: {filename}")

            try:
                with h5py.File(hdf5_filepath, 'r') as f:
                    # Iterate through all groups (sequence lengths) in this file
                    for group_name in f.keys():
                        if group_name.startswith('length_'):
                            seq_len_key = group_name.replace('length_', '')

                            # Skip if filtering is enabled and this sequence length is not in the filter
                            if filter_keys and seq_len_key not in filter_keys:
                                continue

                            group = f[group_name]
                            num_samples = group.attrs['num_samples']

                            # Store: (filepath, group_name, num_samples)
                            file_metadata[seq_len_key].append((hdf5_filepath, group_name, num_samples))
                            print(f"  Found group '{group_name}' with {num_samples} samples")

            except (OSError, KeyError) as e:
                print(f"  WARNING: Error reading HDF5 file: {filename}")
                print(f"  Error: {e}")
                print(f"  Skipping this file.")

    # Print summary
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    for seq_len_key in sorted(file_metadata.keys(), key=int):
        total_samples = sum(count for _, _, count in file_metadata[seq_len_key])
        print(f"Sequence length {seq_len_key}: {total_samples} total samples from {len(file_metadata[seq_len_key])} file(s)")

    print(f"\nTotal sequence length keys: {len(file_metadata)}")

    # =============================================================================
    # 5. TRAIN/VALIDATION SPLIT
    # =============================================================================

    print("\n" + "=" * 80)
    print("5. CREATING TRAIN/VALIDATION SPLIT")
    print("=" * 80)

    train_ratio = 0.8
    val_ratio = 0.2

    train_data = {}
    val_data = {}

    for key in file_metadata.keys():
        print(f"\nSplitting data for sequence length: {key}")

        # Build global indices: list of (file_idx, sample_idx_within_file)
        indices = []
        for file_idx, (filepath, group_name, num_samples) in enumerate(file_metadata[key]):
            for sample_idx in range(num_samples):
                indices.append((file_idx, sample_idx))

        total_samples = len(indices)
        print(f"  Total samples: {total_samples}")

        # Shuffle with fixed seed for reproducibility
        random.shuffle(indices)

        # Split
        train_size = int(train_ratio * total_samples)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}")

        # Store file metadata and corresponding indices
        train_data[key] = (file_metadata[key], train_indices)
        val_data[key] = (file_metadata[key], val_indices)

    # =============================================================================
    # 6. CREATE DATALOADERS
    # =============================================================================

    print("\n" + "=" * 80)
    print("6. CREATING DATALOADERS")
    print("=" * 80)

    train_loaders = create_train_dataloader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loaders = create_train_dataloader(val_data, batch_size=VAL_BATCH_SIZE, shuffle=False)

    print(f"Created {len(train_loaders)} training dataloaders")
    print(f"Created {len(val_loaders)} validation dataloaders")

    # =============================================================================
    # 7. OPTIMIZER AND SCHEDULER
    # =============================================================================

    print("\n" + "=" * 80)
    print("7. SETTING UP OPTIMIZER AND SCHEDULER")
    print("=" * 80)

    TOTAL_STEPS = sum(len(loader) for loader in train_loaders.values()) * CONTINUE_EPOCHS

    print(f"Total training steps: {TOTAL_STEPS}")
    print(f"Learning rate schedule: Cosine decay from {BASE_LR} to {MIN_LR} (no warmup)")

    optimizer = AdamW(
        model.parameters(), lr=BASE_LR, betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY
    )

    def lr_lambda(current_step):
        # Cosine decay from start to end, no warmup
        # Starts at 1.0 (full BASE_LR) and decays to MIN_LR/BASE_LR
        progress = current_step / TOTAL_STEPS
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        return cosine_decay * (BASE_LR - MIN_LR) / BASE_LR + MIN_LR / BASE_LR

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # =============================================================================
    # 8. CONTINUED PRETRAINING LOOP
    # =============================================================================

    print("\n" + "=" * 80)
    print("8. STARTING CONTINUED PRETRAINING")
    print("=" * 80)
    print(f"Epochs: {CONTINUE_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {BASE_LR}")
    print(f"Patch size: {TARGET_PATCH_SIZE}x{TARGET_PATCH_SIZE}")
    print(f"Element length: {TARGET_ELEMENT_LENGTH}")
    print("=" * 80)

    pretrained_model_final = train_lwm(
        model,
        train_loaders,
        val_loaders,
        optimizer,
        scheduler,
        CONTINUE_EPOCHS,
        device=device,
        save_dir=SAVE_DIR,
        log_file=f"continue_pretrain_patch{TARGET_PATCH_SIZE}_log.csv",
        max_batches_per_epoch=1000,
    )

    print("\n" + "=" * 80)
    print("CONTINUED PRETRAINING COMPLETE!")
    print("=" * 80)
    print(f"Models saved to: {SAVE_DIR}")
    print(f"Log file: continue_pretrain_patch{TARGET_PATCH_SIZE}_log.csv")
    print("=" * 80)
