import os
import torch
import torch.nn as nn
import json
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import shutil
from tqdm import tqdm
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Any
import sys
import warnings
import time
warnings.filterwarnings("ignore")
from utils import embedding_space_visual, plot_radar_chart, patch_maker, make_sample, subsample_training_data
from pretrained_model import lwm
from mamba_model import lwm_mamba
from wimae_model import lwm_wimae
import train_heads_config as thc
from collections import defaultdict

# Repo root for resolving paths to data, models, submissions, etc.
_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

# ============================================================
# MODEL SELECTION: Choose which pretrained model to use
# ============================================================
# Set MODEL_TYPE to "transformer", "mamba", or "raw"
# "raw" mode bypasses the pretrained model entirely and uses raw patch features
MODEL_TYPE = "wimae"  # Options: "transformer", "mamba", "raw", "wimae", "contrawimae"

# Fixed patch size for sample variation experiment (will be set by wrapper script)
PATCH_SIZE = 8

# Sample percentages to evaluate (each percentage is tested across all 5 tasks)
SAMPLE_PERCENTAGES = [0.2, 0.4, 0.6, 0.8, 1.0]

# Number of seeds for smoothing tasks with very few training samples (<=10)
N_SMOOTH_SEEDS = 1
SMOOTH_THRESHOLD = 10  # Run multi-seed if expected n_samples <= this

# For model loading compatibility (kept as list for model init)
PATCH_SIZES = [PATCH_SIZE]

# Set the checkpoint path for the selected model
PRETRAINED_CHECKPOINT_PATHS = {
    "mamba": os.path.join(_REPO_ROOT, "outputs/pretrained_models/pretrained_models_mamba_multi_patches/lwm_epoch23_train10630.0000_val13929.3614.pth"),
    "transformer": os.path.join(_REPO_ROOT, "outputs/pretrained_models/pretrained_models_transformer_multi_patches/lwm_epoch39_train23834.3095_val16082.2248.pth"),
    "wimae": os.path.join(_REPO_ROOT, "WirelessContrastiveMaskedLearning/runs/wimae_training_demo/last_checkpoint.pt"),
    "contrawimae": os.path.join(_REPO_ROOT, "WirelessContrastiveMaskedLearning/runs/contrawimae_training_demo/last_checkpoint.pt"),
}
# For "raw" mode, use transformer as dummy backbone (model is not used)
PRETRAINED_CHECKPOINT_PATH = PRETRAINED_CHECKPOINT_PATHS.get(
    "transformer" if MODEL_TYPE == "raw" else MODEL_TYPE,
    PRETRAINED_CHECKPOINT_PATHS["transformer"]
)
# Helper to check if model uses WiMAE-style encoder
IS_WIMAE_STYLE = MODEL_TYPE in ("wimae", "contrawimae")
# ============================================================

# ============================================================
# CUSTOM TOKENIZER WITH CONFIGURABLE PATCH SIZE
# ============================================================
def tokenizer_custom(channels, patch_size=4, max_len=513, masking_percent=0.40, mask=False, seed=42):
    """
    Tokenize wireless channel data into patches with configurable patch size.
    Based on utils.tokenizer but with configurable patch_size parameter.

    Args:
        channels (torch.Tensor or list): Input channel data to be tokenized.
        patch_size (int): Size of square patches (e.g., 4 for 4x4 patches). Defaults to 4.
        max_len (int): Maximum sequence length for tokenized samples. Defaults to 513.
        masking_percent (float): Percentage of patches to mask if mask is True. Defaults to 0.40.
        mask (bool): Whether to apply masking to the tokenized samples. Defaults to False.
        seed (int): Random seed for reproducibility in masking. Defaults to 42.

    Returns:
        dict or torch.Tensor: If mask is True, returns a dictionary mapping sequence lengths to lists
            of tokenized samples. If mask is False, returns a tensor of stacked tokenized samples.
    """
    patches = patch_maker(channels, patch_rows=patch_size, patch_cols=patch_size)
    print(f"\nTokenizing with {patch_size}x{patch_size} patches")
    print("Total number of samples:", len(patches))

    grouped_data = defaultdict(list)  # Group samples by sequence length
    grouped_data_2 = []

    for user_idx in tqdm(range(len(patches)), desc="Processing items"):
        patch_dim = patches[user_idx].shape[1]
        n_patches = patches[user_idx].shape[0]
        n_masks_half = int(masking_percent * n_patches)

        word2id = {
            '[CLS]': 0.2 * np.ones((patch_dim)),
            '[MASK]': 0.1 * np.ones((patch_dim))
        }

        sample = make_sample(
            user_idx, patches, word2id, n_patches, n_masks_half, patch_dim, mask=mask, seed=seed
        )

        if mask:
            seq_length = len(sample[0])
            grouped_data[seq_length].append(sample)
        else:
            grouped_data_2.append(sample)

    if mask:
        normalized_grouped_data = {key: grouped_data[key] for key in sorted(grouped_data.keys())}
    else:
        normalized_grouped_data = torch.stack(grouped_data_2, dim=0).float()

    return normalized_grouped_data


def tokenizer_wimae(channels):
    """
    Resize complex channels to 32x32 and stack real/imag as (N, 2, 32, 32) float.

    Args:
        channels: Complex tensor of shape (N, H, W) or real tensor of shape (N, 2, H, W).

    Returns:
        Float tensor of shape (N, 2, 32, 32).
    """
    import torch.nn.functional as F

    if torch.is_complex(channels):
        real = channels.real.float()
        imag = channels.imag.float()
        x = torch.stack([real, imag], dim=1)  # (N, 2, H, W)
    elif channels.dim() == 4 and channels.shape[1] == 2:
        x = channels.float()
    else:
        raise ValueError(f"Unexpected channel shape: {channels.shape}")

    if x.shape[2] != 32 or x.shape[3] != 32:
        x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)

    return x
# ============================================================

# Set environment variable for CuBLAS deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# Worker initialization for DataLoader to ensure reproducible shuffling
def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)

# List of TaskHeads
task_heads = [
    thc.LosNlosClassificationHead,
    thc.BeamPredictionHead,
    thc.ChannelInterpolationHead,
    thc.ChannelEstimationHead,
    thc.ChannelChartingHead
]

# Fine-tuning wrapper for LWM and downstream model
class FineTuningWrapper(nn.Module):
    def __init__(self, model, task_head, fine_tune_layers="full"):
        """
        Initialize the FineTuningWrapper to manage fine-tuning of a model with a task-specific head.

        Args:
            model (nn.Module): The base model (e.g., LWM) to be fine-tuned.
            task_head (nn.Module): The task-specific head for downstream tasks.
            fine_tune_layers (str or list, optional): Specifies which layers to fine-tune.
                If "full", all model layers are unfrozen. If a list, only specified layers are unfrozen.
                Defaults to "full".

        Raises:
            ValueError: If a specified layer in fine_tune_layers is not found in the model.
        """
        super().__init__()
        self.model = model
        self.task_head = task_head
        
        # Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Handle fine-tuning layers
        if fine_tune_layers is not None:
            if fine_tune_layers == "full":
                # Unfreeze all layers if "full" is specified
                for param in self.model.parameters():
                    param.requires_grad = True
            else:
                # Get a list of all available layer names in the model
                available_layers = [name for name, _ in self.model.named_parameters()]
                
                # Validate that specified layers exist in the model
                for layer in fine_tune_layers:
                    if not any(layer in lname for lname in available_layers):
                        raise ValueError(
                            f"Layer '{layer}' not found in the model. "
                            f"Available layers: {available_layers}"
                        )
                
                # Unfreeze only the specified layers
                for name, param in self.model.named_parameters():
                    if any(layer in name for layer in fine_tune_layers):
                        param.requires_grad = True

    def forward(self, x, input_type="cls_emb", selected_tokens=None):
        """
        Forward pass through the model and task head, processing input based on specified type.

        Args:
            x (torch.Tensor): Input tensor to the model.
            input_type (str, optional): Type of embedding to extract from the model.
                Options: "raw", "cls_emb", "channel_emb", "combined", "mean_pooled",
                "arbitrary_concat", "arbitrary_meanPooled". Defaults to "cls_emb".
            selected_tokens (list, optional): List of token indices for "arbitrary_concat"
                or "arbitrary_meanPooled" input types. Defaults to None.

        Returns:
            torch.Tensor: Output of the task head after processing the input embeddings.
        """
        if input_type == "raw":
            # Mean pool raw patches (similar to mean_pooled for embeddings)
            # This keeps parameter counts comparable to the embedding baseline
            task_input = torch.mean(x, dim=1).unsqueeze(1)

        elif input_type == "raw_channel":
            # Use raw patches without LWM, excluding CLS token (similar to channel_emb)
            # Used for reconstruction tasks that need spatial patch information
            task_input = x[:, 1:]

        else:
            # Pass input through the LWM model to obtain transformer embeddings
            embeddings = self.model(x)
        
            if input_type == "cls_emb":
                # Extract only the [CLS] token embedding (assumed to be at index 0)
                task_input = embeddings[:, [0]]
        
            elif input_type == "channel_emb":
                # Use all patch embeddings except the [CLS] token
                task_input = embeddings[:, 1:]
        
            elif input_type == "combined":
                # Concatenate [CLS] and patch embeddings for full representation
                task_input = embeddings
        
            elif input_type == "mean_pooled":
                # Compute the mean over all token embeddings and retain sequence dimension
                task_input = torch.mean(embeddings, dim=1).unsqueeze(1)
        
            elif input_type == "arbitrary_concat":
                # Concatenate a selected subset of token embeddings by index
                # `selected_tokens` should be a list of token indices to include
                task_input = embeddings[:, selected_tokens]
        
            elif input_type == "arbitrary_meanPooled":
                # Compute mean-pooled embedding over a selected subset of tokens
                # and add a singleton sequence dimension
                task_input = torch.mean(embeddings[:, selected_tokens], dim=1).unsqueeze(1)

        return self.task_head(task_input)

def nmse(y_true, y_pred):
    """
    Calculate the Normalized Mean Squared Error (NMSE) between true and predicted values.

    Args:
        y_true (array-like): Ground truth values.
        y_pred (array-like): Predicted values.

    Returns:
        float: The NMSE value, computed as the mean squared error divided by the mean
               squared magnitude of the true values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred)**2) / np.mean(np.abs(y_true)**2)

def pow2db(nmse):
    """
    Convert a Normalized Mean Squared Error (NMSE) value to decibels (dB).

    Args:
        nmse (float): The NMSE value to convert.

    Returns:
        float: The NMSE value in decibels, calculated as 10 * log10(nmse).
    """
    return 10 * np.log10(nmse)

def finetune(
    base_model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    input_type: str = "cls_emb",
    fine_tune_layers: Optional[str] = None,
    optimizer_config: Optional[Dict[str, Any]] = None,
    scheduler_config: Optional[Dict[str, Any]] = None,
    epochs: int = 50,
    device: str = "cuda",
    task: Optional[str] = None,
    d_model: Optional[int] = None,
    sequence_length: Optional[int] = None,
    selected_tokens: Optional[List[int]] = None,
    bbox_coord: Optional[float] = None,
    max_head_pars: int = 1e5,
    max_wrapper_pars: int = 3e6,
    patch_size: int = 4,
) -> Tuple[nn.Module, List[float], List[float], List[float], List[float], List[torch.Tensor], List[torch.Tensor]]:
    """
    Fine-tune a pre-trained base model with a task-specific head on a given dataset.

    Args:
        base_model (nn.Module): Pre-trained base model (e.g., LWM) to fine-tune.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (Optional[DataLoader]): DataLoader for the validation dataset. Defaults to None.
        test_loader (Optional[DataLoader]): DataLoader for the test dataset. Defaults to None.
        input_type (str): Type of input embedding to use. Options: 'cls_emb', 'mean_pooled',
            'channel_emb', 'combined', 'arbitrary_meanPooled'. Defaults to 'cls_emb'.
        fine_tune_layers (Optional[str]): Layers to fine-tune in the base model. If 'full', all
            layers are fine-tuned; if a list, only specified layers are fine-tuned. Defaults to None.
        optimizer_config (Optional[Dict[str, Any]]): Configuration for the optimizer.
            Defaults to {'lr': 1e-3} if None.
        scheduler_config (Optional[Dict[str, Any]]): Configuration for the learning rate scheduler.
            Defaults to {'step_size': 1000, 'gamma': 0.99} if None.
        epochs (int): Number of training epochs. Defaults to 50.
        device (str): Device for training ('cuda' or 'cpu'). Defaults to 'cuda'.
        task (Optional[str]): Task name. Options: 'LosNlosClassification', 'BeamPrediction',
            'ChannelInterpolation', 'ChannelEstimation', 'ChannelCharting'. Defaults to None.
        d_model (Optional[int]): Dimensionality of the model embeddings. Required.
        sequence_length (Optional[int]): Length of the input sequence. Required for
            'channel_emb' or 'combined' input types.
        selected_tokens (Optional[List[int]]): List of token indices for 'arbitrary_meanPooled'
            or 'arbitrary_concat' input types. Defaults to None.
        bbox_coord (Optional[float]): Bounding box coordinate (not used in the function).
            Defaults to None.
        max_head_pars (int): Maximum allowed parameters in the task head. Defaults to 100,000.
        max_wrapper_pars (int): Maximum allowed parameters in the wrapper. Defaults to 3,000,000.

    Returns:
        Tuple containing:
            - nn.Module: Fine-tuned wrapper model.
            - List[float]: Training losses per epoch.
            - List[float]: Validation losses per epoch.
            - List[float]: Test loss (single value) after training.
            - List[float]: Task-specific score (e.g., F1-score or normalized score).
            - List[torch.Tensor]: Ground truth labels from the test set.
            - List[torch.Tensor]: Predictions from the test set.

    Raises:
        ValueError: If task, d_model, or input_type is invalid, or required parameters
            (e.g., sequence_length, selected_tokens) are missing.
    """
    # Validate inputs
    if task is None or d_model is None:
        raise ValueError("Task and d_model must be provided.")
    if input_type not in ["cls_emb", "mean_pooled", "channel_emb", "combined", "arbitrary_meanPooled", "raw", "raw_channel"]:
        raise ValueError(f"Invalid input_type: {input_type}")

    # Determine number of patches based on input type
    # Note: "raw" uses mean pooling, so n_patches=1 (same as mean_pooled)
    # Note: "raw_channel" uses full sequence minus CLS (same as channel_emb)
    if input_type in ["cls_emb", "mean_pooled", "arbitrary_meanPooled", "raw"]:
        n_patches = 1
    elif input_type in ["channel_emb", "raw_channel"]:
        if sequence_length is None:
            raise ValueError(f"sequence_length must be provided for input_type '{input_type}'.")
        n_patches = sequence_length - 1
    elif input_type == "combined":
        if sequence_length is None:
            raise ValueError("sequence_length must be provided for input_type 'combined'.")
        n_patches = sequence_length
    else:  # arbitrary_meanPooled
        if selected_tokens is None:
            raise ValueError("selected_tokens must be provided for input_type 'arbitrary_meanPooled'.")
        n_patches = len(selected_tokens)

    # Define input dimension
    # For "raw" and "raw_channel" input, use patch_dim (element_length) instead of d_model
    if input_type in ["raw", "raw_channel"]:
        element_length = patch_size * patch_size * 2
        input_dim = (n_patches, element_length)
    else:
        input_dim = (n_patches, d_model)

    # Dynamically determine output_dim for regression tasks
    output_dim = None
    if task in ["ChannelInterpolation", "ChannelEstimation"]:
        for batch in train_loader:
            output_dim = batch[1].shape[1:]
            break  # Use the first batch to determine output shape

    # Handle DataParallel models
    if isinstance(base_model, nn.DataParallel):
        base_model = base_model.module

    # Initialize task-specific head
    if task == "LosNlosClassification":
        task_head = thc.LosNlosClassificationHead(input_dim)
    elif task == "BeamPrediction":
        task_head = thc.BeamPredictionHead(input_dim)
    elif task == "ChannelInterpolation":
        if output_dim is None:
            raise ValueError("output_dim could not be determined for ChannelInterpolation.")
        if IS_WIMAE_STYLE:
            task_head = thc.WiMAEChannelReconstructionHead(input_dim, output_dim)
        else:
            task_head = thc.ChannelInterpolationHead(input_dim, output_dim, patch_size=PATCH_SIZE)
    elif task == "ChannelEstimation":
        if output_dim is None:
            raise ValueError("output_dim could not be determined for ChannelEstimation.")
        if IS_WIMAE_STYLE:
            task_head = thc.WiMAEChannelReconstructionHead(input_dim, output_dim)
        else:
            task_head = thc.ChannelEstimationHead(input_dim, output_dim, patch_size=PATCH_SIZE)
    elif task == "ChannelCharting":
        task_head = thc.ChannelChartingHead(input_dim)
    else:
        raise ValueError(f"Unsupported task: {task}")

    # Set up loss criterion
    if task in ["LosNlosClassification", "BeamPrediction"]:
        criterion = nn.CrossEntropyLoss()
    elif task in ["ChannelInterpolation", "ChannelEstimation", "ChannelCharting"]:
        criterion = nn.MSELoss()

    # Initialize the fine-tuning wrapper
    fine_tune_layers_config = None if task == "LosNlosClassification" else fine_tune_layers
    wrapper = FineTuningWrapper(
        model=base_model,
        task_head=task_head,
        fine_tune_layers=fine_tune_layers_config
    )
    wrapper = wrapper.to(device)
    
    n_head_pars = count_parameters(wrapper.task_head)
    n_wrapper_pars = count_parameters(wrapper)
    print(f"\nNumber of head parameters:    {n_head_pars}")
    print(f"Number of wrapper parameters: {n_wrapper_pars}\n")
    if n_head_pars > max_head_pars or n_wrapper_pars > max_wrapper_pars:
        reasons = []
        if n_head_pars > max_head_pars:
            reasons.append(
                f"head parameters ({n_head_pars}) exceed maximum allowed ({max_head_pars})"
            )
        if n_wrapper_pars > max_wrapper_pars:
            reasons.append(
                f"wrapper parameters ({n_wrapper_pars}) exceed maximum allowed ({max_wrapper_pars})"
            )
        print("Stopping run because " + " and ".join(reasons))
        sys.exit(1)
        
    # Save universal LWM weights
    os.makedirs("submission", exist_ok=True)
    torch.save(base_model.state_dict(), os.path.join(_REPO_ROOT, "outputs/submissions/submission/model_checkpoint.pth"))
    shutil.copy("pretrained_model.py", "submission/pretrained_model.py")
    shutil.copy("utils.py", "submission/utils.py")
    shutil.copy("train_heads_config.py", "submission/train_heads_config.py")
    shutil.copy("train_heads.py", "submission/train_heads.py")
    
    # Set default optimizer config if not provided
    if optimizer_config is None:
        optimizer_config = {"lr": 1e-3}
    optimizer = torch.optim.Adam(wrapper.parameters(), **optimizer_config)
    
    # Set up the scheduler
    if scheduler_config is None:
        scheduler_config = {"step_size": 1000, "gamma": 0.99}
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_config["step_size"],
        gamma=scheduler_config["gamma"]
    )
    
    # Initialize training utilities
    train_losses, val_losses, f1_scores = [], [], []
    predictions, ground_truth = [], []
    
    # Training loop
    for epoch in range(epochs):
        wrapper.train()
        epoch_loss = 0.0
        batch_count = 0
        train_preds, train_targets = [], []
        
        # Prepare a single validation batch
        val_batch = None
        val_iterator = iter(val_loader) if val_loader else None
        if val_iterator:
            try:
                val_batch = next(val_iterator)
            except StopIteration:
                val_iterator = None
                
        with tqdm(train_loader, desc=f"Task Epoch {epoch + 1}/{epochs}", leave=True) as progress_bar:
            for batch in progress_bar:
                input_data, targets = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
    
                outputs = wrapper(input_data, 
                                input_type=input_type,
                                selected_tokens=selected_tokens)
                if task in ["LosNlosClassification", "BeamPrediction"]:
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    train_preds.extend(preds)
                    train_targets.extend(targets.cpu().numpy())
                elif task in ["ChannelInterpolation", "ChannelEstimation"]:
                    train_preds.extend(outputs.cpu().detach().numpy().flatten())
                    train_targets.extend(targets.cpu().detach().numpy().flatten())
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                running_avg_loss = epoch_loss / batch_count
                
                train_metric = None
                if task in ["LosNlosClassification", "BeamPrediction"] and train_preds and train_targets:
                    train_metric = f1_score(train_targets, train_preds, average="weighted")
                elif task in ["ChannelInterpolation", "ChannelEstimation"] and train_preds and train_targets:
                    train_metric = nmse(train_targets, train_preds)
    
                val_loss = 0.0
                val_preds, val_targets = [], []
                if val_batch:
                    wrapper.eval()
                    with torch.no_grad():
                        val_input_data, val_targets_batch = val_batch[0].to(device), val_batch[1].to(device)
                        val_outputs = wrapper(val_input_data, input_type=input_type)
                        if task in ["LosNlosClassification", "BeamPrediction"]:
                            val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
                            val_targets = val_targets_batch.cpu().numpy()
                        elif task in ["ChannelInterpolation", "ChannelEstimation"]:
                            val_preds = val_outputs.cpu().numpy().flatten()
                            val_targets = val_targets_batch.cpu().numpy().flatten()
                        val_loss = criterion(val_outputs, val_targets_batch).item()
    
                    avg_val_loss = val_loss if val_loss > 0 else None
                    val_metric = None
                    if task in ["LosNlosClassification", "BeamPrediction"] and len(val_preds) and len(val_targets):
                        val_metric = f1_score(val_targets, val_preds, average="weighted")
                    elif task in ["ChannelInterpolation", "ChannelEstimation"] and len(val_preds) and len(val_targets):
                        val_metric = nmse(val_targets, val_preds)
                    
                    # Switch back to training mode for the next batch
                    wrapper.train()
    
                postfix_dict = {
                    "Batch Loss": f"{loss.item():.6f}",
                    "Avg Train Loss": f"{running_avg_loss:.6f}",
                }
                if train_metric is not None:
                    if task in ["LosNlosClassification", "BeamPrediction"]:
                        postfix_dict["Train F1-Score"] = f"{train_metric:.4f}"
                    elif task in ["ChannelInterpolation", "ChannelEstimation"]:
                        postfix_dict["Train NMSE"] = f"{pow2db(train_metric):.6f}"
                if avg_val_loss is not None:
                    postfix_dict["Avg Val Loss"] = f"{avg_val_loss:.6f}"
                if val_metric is not None:
                    if task in ["LosNlosClassification", "BeamPrediction"]:
                        postfix_dict["Val F1-Score"] = f"{val_metric:.4f}"
                    elif task in ["ChannelInterpolation", "ChannelEstimation"]:
                        postfix_dict["Val NMSE"] = f"{pow2db(val_metric):.6f}"
    
                progress_bar.set_postfix(postfix_dict)
                progress_bar.refresh()
    
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
    
            train_metric = None
            if task in ["LosNlosClassification", "BeamPrediction"] and train_preds and train_targets:
                train_metric = f1_score(train_targets, train_preds, average="weighted")
            elif task in ["ChannelInterpolation", "ChannelEstimation"] and train_preds and train_targets:
                train_metric = nmse(train_targets, train_preds)
    
            val_loss = 0.0
            val_preds, val_targets = [], []
            if val_loader:
                wrapper.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        input_data, targets = batch[0].to(device), batch[1].to(device)
                        outputs = wrapper(input_data, input_type=input_type)
                        if task in ["LosNlosClassification", "BeamPrediction"]:
                            preds = torch.argmax(outputs, dim=1).cpu().numpy()
                            val_preds.extend(preds)
                            val_targets.extend(targets.cpu().numpy())
                        elif task in ["ChannelInterpolation", "ChannelEstimation"]:
                            val_preds.extend(outputs.cpu().numpy().flatten())
                            val_targets.extend(targets.cpu().numpy().flatten())
                        elif task == "ChannelCharting":
                            val_preds.extend(outputs.cpu().numpy().flatten())
                            val_targets.extend(targets.cpu().numpy().flatten())
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
    
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
    
                val_metric = None
                if task in ["LosNlosClassification", "BeamPrediction"] and val_preds and val_targets:
                    val_metric = f1_score(val_targets, val_preds, average="weighted")
                    f1_scores.append(val_metric)
                elif task in ["ChannelInterpolation", "ChannelEstimation"] and val_preds and val_targets:
                    val_metric = nmse(val_targets, val_preds)
                elif task == "ChannelCharting" and val_preds and val_targets:
                    val_metric = np.mean(np.abs(np.array(val_targets) - np.array(val_preds)))
    
                if val_metric is not None and task == "ChannelCharting":
                    print(f"Validation Prediction Error (meters) at epoch {epoch + 1}: {val_metric:.2f}")
    
            postfix_dict = {
                "Avg Train Loss": f"{avg_train_loss:.6f}",
            }
            if train_metric is not None:
                if task in ["LosNlosClassification", "BeamPrediction"]:
                    postfix_dict["Train F1-Score"] = f"{train_metric:.4f}"
                elif task in ["ChannelInterpolation", "ChannelEstimation"]:
                    postfix_dict["Train NMSE"] = f"{pow2db(train_metric):.6f}"
            if avg_val_loss is not None:
                postfix_dict["Avg Val Loss"] = f"{avg_val_loss:.6f}"
            if val_metric is not None:
                if task in ["LosNlosClassification", "BeamPrediction"]:
                    postfix_dict["Val F1-Score"] = f"{val_metric:.4f}"
                elif task in ["ChannelInterpolation", "ChannelEstimation"]:
                    postfix_dict["Val NMSE"] = f"{pow2db(val_metric):.6f}"
    
            progress_bar.set_postfix(postfix_dict)
            progress_bar.refresh()
    
        scheduler.step()

    # Test evaluation
    test_loss = 0.0
    test_preds, test_targets = [], []
    if test_loader:
        wrapper.eval()
        with torch.no_grad():
            for batch in test_loader:
                input_data, targets = batch[0].to(device), batch[1].to(device)
                outputs = wrapper(input_data, input_type=input_type)
                if epoch == epochs - 1:
                    predictions.append(outputs)
                    ground_truth.append(targets)
                if task in ["LosNlosClassification", "BeamPrediction"]:
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    test_preds.extend(preds)
                    test_targets.extend(targets.cpu().numpy())
                elif task in ["ChannelInterpolation", "ChannelEstimation"]:
                    test_preds.extend(outputs.cpu().numpy().flatten())
                    test_targets.extend(targets.cpu().numpy().flatten())
                elif task == "ChannelCharting":
                    test_preds.extend(outputs.cpu().numpy().flatten())
                    test_targets.extend(targets.cpu().numpy().flatten())
                loss = criterion(outputs, targets)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        
        test_metric = None
        if task in ["LosNlosClassification", "BeamPrediction"] and test_preds and test_targets:
            test_metric = f1_score(test_targets, test_preds, average="weighted")
        elif task in ["ChannelInterpolation", "ChannelEstimation"] and test_preds and test_targets:
            test_metric = nmse(test_targets, test_preds)
        elif task == "ChannelCharting" and test_preds and test_targets:
            test_metric = np.mean(np.abs(np.array(test_targets) - np.array(test_preds)))
            
        print(f"Test Loss: {avg_test_loss:.6f}")
        if test_metric is not None:
            if task in ["LosNlosClassification", "BeamPrediction"]:
                print(f"Test F1-Score: {test_metric:.4f}")
            elif task in ["ChannelInterpolation", "ChannelEstimation"]:
                print(f"Test NMSE (dB): {pow2db(test_metric):.6f}")
            elif task == "ChannelCharting":
                print(f"Test Prediction Error (meters): {test_metric:.2f}")
                
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    if val_losses:
        plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curves")
    plt.legend()
    plt.grid(True)
    plt.show()

    test_losses = [avg_test_loss] if test_loader else []

    if task in ["LosNlosClassification", "BeamPrediction"]:
        score = test_metric
    elif task in ["ChannelInterpolation", "ChannelEstimation"]:
        db_value = pow2db(test_metric)
        db_min, db_max = -20.0, 0.0
        normalized = (db_value - db_min) / (db_max - db_min)
        score = 1.0 - normalized
        score = max(0.0, min(1.0, score))
    elif task == "ChannelCharting":
        localization_error = max(0.0, min(100.0, test_metric))
        score = (100.0 - localization_error) / 100.0
    
    print("\n=============================================================")
    print(f"The score for the {task} task is {score:.5f}")
    print("=============================================================\n")
    
    return wrapper, train_losses, val_losses, test_losses, score, ground_truth, predictions

def count_parameters(model):
    """
    Calculate the total number of learnable parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to count parameters for.

    Returns:
        int: The total number of parameters that require gradients.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_latency(
    wrapper: nn.Module,
    test_loader: DataLoader,
    input_type: str = "cls_emb",
    selected_tokens: Optional[List[int]] = None,
    device: str = "cuda",
    warmup_runs: int = 10,
    measure_runs: int = 100
) -> Dict[str, Any]:
    """
    Measure inference latency for a given model wrapper.

    Args:
        wrapper (nn.Module): The fine-tuned wrapper model to measure.
        test_loader (DataLoader): DataLoader for the test dataset.
        input_type (str): Type of input embedding to use.
        selected_tokens (Optional[List[int]]): List of token indices if needed.
        device (str): Device for inference ('cuda' or 'cpu').
        warmup_runs (int): Number of warmup iterations before measurement.
        measure_runs (int): Number of iterations to measure and average.

    Returns:
        Dict[str, Any]: Dictionary containing latency statistics in milliseconds.
    """
    wrapper.eval()

    # Get a single batch for measurement
    test_batch = None
    for batch in test_loader:
        test_batch = batch
        break

    if test_batch is None:
        return {
            "mean_latency_ms": None,
            "std_latency_ms": None,
            "min_latency_ms": None,
            "max_latency_ms": None,
            "median_latency_ms": None
        }

    input_data = test_batch[0].to(device)
    batch_size = input_data.size(0)

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = wrapper(input_data, input_type=input_type, selected_tokens=selected_tokens)

    # Synchronize before measurement
    if device == "cuda":
        torch.cuda.synchronize()

    # Measure latency
    latencies = []
    with torch.no_grad():
        for _ in range(measure_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            _ = wrapper(input_data, input_type=input_type, selected_tokens=selected_tokens)

            if device == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds

    latencies = np.array(latencies)

    return {
        "mean_latency_ms": float(np.mean(latencies)),
        "std_latency_ms": float(np.std(latencies)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "batch_size": int(batch_size),
        "per_sample_mean_latency_ms": float(np.mean(latencies) / batch_size)
    }

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model once (for mamba with multi-patch, all resolutions are built-in)
# For "raw" mode, use transformer as dummy backbone (model is bypassed in forward pass)
if MODEL_TYPE == "raw":
    print(f"\nRAW MODE: Using transformer backbone (model will be bypassed)")
else:
    print(f"\nLoading {MODEL_TYPE.upper()} model from {PRETRAINED_CHECKPOINT_PATH}")

if MODEL_TYPE == "mamba":
    universal_lwm = lwm_mamba(
        element_length=32,  # overridden internally when patch_sizes is set
        d_model=128,
        n_layers=12,
        max_len=513,
        d_state=8,
        d_conv=4,
        expand=1.2,
        dropout=0.1,
        bidirectional=True,
        patch_sizes=PATCH_SIZES
    ).to(device)

    checkpoint = torch.load(PRETRAINED_CHECKPOINT_PATH, map_location=device)
    clean_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    # strict=False to ignore checkpoint weights for patch sizes not in PATCH_SIZES
    universal_lwm.load_state_dict(clean_state_dict, strict=False)

elif IS_WIMAE_STYLE:
    universal_lwm = lwm_wimae(
        checkpoint_path=PRETRAINED_CHECKPOINT_PATH,
        device=device
    ).to(device)
    # Save initial state dict so we can reset after each task's fine-tuning
    clean_state_dict = {k: v.clone() for k, v in universal_lwm.state_dict().items()}

elif MODEL_TYPE == "transformer" or MODEL_TYPE == "raw":
    universal_lwm = lwm(
        element_length=32,  # ignored when patch_sizes is set
        d_model=128,
        n_layers=12,
        max_len=513,
        n_heads=8,
        dropout=0.1,
        patch_sizes=PATCH_SIZES
    ).to(device)

    # Skip loading weights for "raw" mode (model is not used)
    if MODEL_TYPE == "transformer":
        checkpoint = torch.load(PRETRAINED_CHECKPOINT_PATH, map_location=device)
        clean_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        # strict=False to ignore checkpoint weights for patch sizes not in PATCH_SIZES
        universal_lwm.load_state_dict(clean_state_dict, strict=False)
    else:
        clean_state_dict = None  # No weights to reload for raw mode

# Override training configs for "raw" mode
# Use raw_channel for all tasks (flattened raw patch features, no LWM)
if MODEL_TYPE == "raw":
    for cfg in thc.training_configs:
        cfg["input_type"] = "raw_channel"
        cfg["fine_tune_layers"] = None  # LWM is bypassed

# Override training configs for "wimae" mode
# No CLS token: remap cls_emb → mean_pooled, channel_emb → combined
if IS_WIMAE_STYLE:
    for cfg in thc.training_configs:
        if cfg["input_type"] == "cls_emb":
            cfg["input_type"] = "mean_pooled"
        elif cfg["input_type"] == "channel_emb":
            cfg["input_type"] = "combined"
        cfg["fine_tune_layers"] = ["layers.9", "layers.10", "layers.11"]

# Results storage
percentage_results = {}
num_tasks = 5
task_names = ["LoS/NLoS\nClassification", "Beam\nPrediction", "Channel\nInterpolation", "Channel\nEstimation", "User\nLocalization"]
baseline_scores = [0.9396, 0.6137, 0.4165, 0.4576, 0.6711]

for sample_pct in SAMPLE_PERCENTAGES:
    print(f"\n{'='*70}")
    print(f" Evaluating with {int(sample_pct*100)}% of training data")
    print(f"{'='*70}")

    scores = []
    sample_counts = {}  # Track actual sample sizes per task
    pct_start_time = time.time()

    for t in range(1, num_tasks + 1):
        # Load task configuration
        task_data_dir = os.path.join(_REPO_ROOT, f"task_{t}")
        os.makedirs(task_data_dir, exist_ok=True)

        with open(f"{task_data_dir}/config.json", "r") as f:
            config = json.load(f)

        # Load data
        train_data = torch.load(f"{task_data_dir}/train_data.pt", map_location="cpu")
        val_data = torch.load(f"{task_data_dir}/val_data.pt", map_location="cpu") if os.path.exists(f"{task_data_dir}/val_data.pt") else None
        test_data = torch.load(f"{task_data_dir}/test_data.pt", map_location="cpu") if os.path.exists(f"{task_data_dir}/test_data.pt") else None

        # Retrieve training configuration
        training_config = thc.training_configs[t-1]
        task_name = training_config['task']
        seed = training_config["seed"]

        print(f"Starting Task {t}/{num_tasks}: {task_name}")
        print(f"  Sample percentage: {int(sample_pct*100)}%")

        # Determine if we need multi-seed smoothing for small-sample tasks
        total_train_size = train_data["channels"].shape[0]
        expected_n_samples = max(2, int(total_train_size * sample_pct))
        n_smooth_runs = N_SMOOTH_SEEDS if expected_n_samples <= SMOOTH_THRESHOLD else 1
        if n_smooth_runs > 1:
            print(f"  Multi-seed smoothing: {n_smooth_runs} runs (expected {expected_n_samples} samples)")

        smooth_scores = []
        for smooth_idx in range(n_smooth_runs):
            run_seed = seed + smooth_idx

            # Set random seed for reproducibility
            torch.manual_seed(run_seed)
            np.random.seed(run_seed)
            torch.cuda.manual_seed_all(run_seed)

            # Reload pretrained weights to reset any fine-tuning changes
            if clean_state_dict is not None:
                universal_lwm.load_state_dict(clean_state_dict, strict=False)
            universal_lwm.requires_grad_(False)

            # Subsample training data
            train_data_subsampled, n_train_samples = subsample_training_data(
                train_data,
                sample_pct,
                seed=run_seed,
                task_idx=t
            )

            # Extract channels and labels from SUBSAMPLED data
            train_channels = train_data_subsampled["channels"]
            val_channels = val_data["channels"] if val_data else None
            test_channels = test_data["channels"] if test_data else None
            if t <= 2:
                train_labels = train_data_subsampled["labels"].to(device).long()
                val_labels = val_data["labels"].to(device).long() if val_data else None
                test_labels = test_data["labels"].to(device).long() if test_data else None
            else:
                train_labels = train_data_subsampled["labels"].to(device)
                val_labels = val_data["labels"].to(device) if val_data else None
                test_labels = test_data["labels"].to(device) if test_data else None

            # Tokenize input data
            if IS_WIMAE_STYLE:
                train_tokens = tokenizer_wimae(train_channels)
                val_tokens = tokenizer_wimae(val_channels) if val_channels is not None else None
                test_tokens = tokenizer_wimae(test_channels) if test_channels is not None else None
            else:
                train_tokens = tokenizer_custom(train_channels, patch_size=PATCH_SIZE)
                val_tokens = tokenizer_custom(val_channels, patch_size=PATCH_SIZE) if val_channels is not None else None
                test_tokens = tokenizer_custom(test_channels, patch_size=PATCH_SIZE) if test_channels is not None else None

            # Determine sequence length
            if IS_WIMAE_STYLE:
                sequence_length = 128
            else:
                sequence_length = train_tokens.shape[1]

            # Handle edge case where subsampled data is smaller than batch_size
            effective_batch_size = min(training_config["batch_size"], max(1, n_train_samples))
            if effective_batch_size < training_config["batch_size"] and smooth_idx == 0:
                print(f"  Adjusted batch size: {training_config['batch_size']} → {effective_batch_size}")

            # Create datasets and data loaders
            train_dataset = TensorDataset(train_tokens, train_labels)
            train_loader = DataLoader(
                train_dataset,
                batch_size=effective_batch_size,
                shuffle=True,
                worker_init_fn=worker_init_fn,
                num_workers=0
            )
            if val_data:
                val_dataset = TensorDataset(val_tokens, val_labels)
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=training_config["batch_size"],
                    shuffle=False,
                    worker_init_fn=worker_init_fn,
                    num_workers=0
                )
            else:
                val_loader = None
            if test_data:
                test_dataset = TensorDataset(test_tokens, test_labels)
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=training_config["batch_size"],
                    shuffle=False,
                    worker_init_fn=worker_init_fn,
                    num_workers=0
                )
            else:
                test_loader = None

            # Fine-tune the model
            wrapper, train_losses, val_losses, test_losses, score, ground_truth, predictions = finetune(
                base_model=universal_lwm,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                input_type=training_config["input_type"],
                fine_tune_layers=training_config["fine_tune_layers"],
                optimizer_config=training_config["optimizer_config"],
                scheduler_config=training_config["scheduler"],
                epochs=training_config["epochs"],
                task=training_config["task"],
                d_model=universal_lwm.d_model,
                sequence_length=sequence_length,
                selected_tokens=training_config["selected_tokens"],
                bbox_coord=config["bounding_box_coord"] if t == 5 else None,
                max_head_pars=5e6,
                max_wrapper_pars=config["max_wrapper_parameters"],
                device=device,
                patch_size=PATCH_SIZE
            )
            smooth_scores.append(float(score))

            if n_smooth_runs > 1:
                print(f"  Seed {smooth_idx+1}/{n_smooth_runs}: score = {float(score):.4f}")

        # Average scores across seeds for smoothing
        if n_smooth_runs > 1:
            score = float(np.mean(smooth_scores))
            print(f"  Smoothed score ({n_smooth_runs} seeds): {score:.4f} (std: {np.std(smooth_scores):.4f})")

        sample_counts[f"task_{t}"] = n_train_samples

        # Create submission directory with sample percentage
        pct_label = f"pct_{int(sample_pct*100)}"
        save_dir = os.path.join(_REPO_ROOT, f"outputs/submissions/submission_sample_variation_{MODEL_TYPE}/{pct_label}/task_{t}")
        os.makedirs(save_dir, exist_ok=True)

        # Save fine-tuned wrapper model weights
        wrapper_weights_path = os.path.join(save_dir, "wrapper.pt")
        torch.save(wrapper.state_dict(), wrapper_weights_path)
        print(f"Saved wrapper weights to {wrapper_weights_path}")

        # Save ground truth and predictions
        torch.save(ground_truth, os.path.join(save_dir, "ground_truth.pt"))
        torch.save(predictions, os.path.join(save_dir, "predictions.pt"))
        print(f"Saved ground truth and predictions for task {t}")

        # Save task score
        score_path = os.path.join(save_dir, "score.json")
        with open(score_path, "w") as f:
            json.dump(float(score), f, indent=2)
        print(f"Saved task score to {score_path}")

        # Measure inference latency
        print(f"\nMeasuring inference latency for task {t}...")
        latency_metrics = measure_inference_latency(
            wrapper=wrapper,
            test_loader=test_loader,
            input_type=training_config["input_type"],
            selected_tokens=training_config["selected_tokens"],
            device=device,
            warmup_runs=10,
            measure_runs=100
        )

        # Collect metadata
        element_length = PATCH_SIZE * PATCH_SIZE * 2
        metadata = {
            "task_name": task_name,
            "model_type": MODEL_TYPE,
            "patch_size": PATCH_SIZE,
            "patch_dim": element_length,
            "sample_percentage": float(sample_pct),
            "n_train_samples": int(n_train_samples),
            "original_train_size": int(train_data["channels"].shape[0]),
            "channel_shape": {
                "train": list(train_channels.shape),
                "val": list(val_channels.shape) if val_channels is not None else None,
                "test": list(test_channels.shape) if test_channels is not None else None
            },
            "token_shape": {
                "train": list(train_tokens.shape),
                "val": list(val_tokens.shape) if val_tokens is not None else None,
                "test": list(test_tokens.shape) if test_tokens is not None else None
            },
            "sequence_length": int(sequence_length),
            "d_model": int(universal_lwm.d_model),
            "input_type": training_config["input_type"],
            "latency_metrics": latency_metrics,
            "model_parameters": {
                "head_parameters": int(count_parameters(wrapper.task_head)),
                "wrapper_parameters": int(count_parameters(wrapper))
            }
        }

        # Save metadata
        metadata_path = os.path.join(save_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved task metadata to {metadata_path}")

        # Print latency summary
        if latency_metrics["mean_latency_ms"] is not None:
            print(f"\nLatency Summary for Task {t}:")
            print(f"  Mean Latency:        {latency_metrics['mean_latency_ms']:.3f} ms (batch size: {latency_metrics['batch_size']})")
            print(f"  Per-Sample Latency:  {latency_metrics['per_sample_mean_latency_ms']:.3f} ms")
            print(f"  Std Dev:             {latency_metrics['std_latency_ms']:.3f} ms")
            print(f"  Min/Max:             {latency_metrics['min_latency_ms']:.3f} / {latency_metrics['max_latency_ms']:.3f} ms")
            print(f"  Median:              {latency_metrics['median_latency_ms']:.3f} ms")

        scores.append(float(score))

    # Calculate time for this percentage
    pct_time = time.time() - pct_start_time
    composite = float(np.mean(scores))

    task_score_dict = {}
    for i, s in enumerate(scores):
        task_score_dict[f"task_{i+1}"] = s

    percentage_results[sample_pct] = {
        "sample_counts": sample_counts,
        "task_scores": task_score_dict,
        "composite_score": composite,
        "training_time_seconds": pct_time
    }

    # Save per-percentage composite score
    pct_composite_path = os.path.join(_REPO_ROOT, f"outputs/submissions/submission_sample_variation_{MODEL_TYPE}/{pct_label}/composite_score.json")
    with open(pct_composite_path, "w") as f:
        json.dump({
            "sample_percentage": int(sample_pct * 100),
            "sample_counts": sample_counts,
            "task_scores": task_score_dict,
            "composite_score": composite
        }, f, indent=2)

    print(f"\nComposite score for {int(sample_pct*100)}% samples: {composite:.5f}")
    print(f"Training time: {pct_time:.1f} seconds")

# Generate Aggregated Results
print("\n" + "="*70)
print(" Generating Aggregated Results")
print("="*70)

# Build results by task view
results_by_task = {}
for t in range(1, num_tasks + 1):
    task_name = thc.training_configs[t-1]["task"]
    results_by_task[f"task_{t}"] = {
        "name": task_name,
        "results": {}
    }
    for pct, pct_data in percentage_results.items():
        results_by_task[f"task_{t}"]["results"][str(int(pct*100))] = {
            "score": pct_data["task_scores"][f"task_{t}"],
            "n_samples": pct_data["sample_counts"][f"task_{t}"]
        }

# Build aggregated results
aggregated_results = {
    "experiment_config": {
        "patch_size": PATCH_SIZE,
        "sample_percentages": [int(p*100) for p in SAMPLE_PERCENTAGES],
        "model_type": MODEL_TYPE,
        "num_tasks": num_tasks
    },
    "results_by_percentage": {
        str(int(pct*100)): {
            "sample_counts": pct_data["sample_counts"],
            "task_scores": pct_data["task_scores"],
            "composite_score": pct_data["composite_score"],
            "training_time_seconds": pct_data["training_time_seconds"]
        }
        for pct, pct_data in percentage_results.items()
    },
    "results_by_task": results_by_task,
    "summary_statistics": {
        "composite_scores": [pct_data["composite_score"] for pct_data in percentage_results.values()],
        "best_percentage": int(max(percentage_results.items(), key=lambda x: x[1]["composite_score"])[0] * 100),
        "best_composite_score": max(pct_data["composite_score"] for pct_data in percentage_results.values())
    }
}

# Save aggregated results
aggregated_path = os.path.join(_REPO_ROOT, f"outputs/submissions/submission_sample_variation_{MODEL_TYPE}/aggregated_results.json")
with open(aggregated_path, "w") as f:
    json.dump(aggregated_results, f, indent=2)

print(f"\nSaved aggregated results to {aggregated_path}")
print(f"Best percentage: {aggregated_results['summary_statistics']['best_percentage']}%")
print(f"Best composite score: {aggregated_results['summary_statistics']['best_composite_score']:.5f}")

# Create zip archive
shutil.make_archive(os.path.join(_REPO_ROOT, f"outputs/submissions/submission_sample_variation_{MODEL_TYPE}"), format="zip", root_dir=os.path.join(_REPO_ROOT, f"outputs/submissions/submission_sample_variation_{MODEL_TYPE}"))
print(f"\nCreated submission_sample_variation_{MODEL_TYPE}.zip")

# Plot radar charts comparing sample percentages
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2 * np.pi, num_tasks, endpoint=False).tolist()
angles += angles[:1]

# Plot baseline
baseline_values = baseline_scores + baseline_scores[:1]
ax.plot(angles, baseline_values, 'k--', linewidth=1, label='Baseline')
ax.fill(angles, baseline_values, alpha=0.05, color='gray')

# Plot each percentage
colors = plt.cm.viridis(np.linspace(0, 1, len(SAMPLE_PERCENTAGES)))
for (pct, pct_data), color in zip(percentage_results.items(), colors):
    scores_list = list(pct_data["task_scores"].values()) + [list(pct_data["task_scores"].values())[0]]
    ax.plot(angles, scores_list, '-o', linewidth=2,
            label=f'{int(pct*100)}% (comp={pct_data["composite_score"]:.4f})', color=color)
    ax.fill(angles, scores_list, alpha=0.1, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(task_names)
ax.set_ylim(0, 1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax.set_title("Performance vs Sample Size")
plt.tight_layout()
plt.savefig(os.path.join(_REPO_ROOT, f"outputs/submissions/submission_sample_variation_{MODEL_TYPE}/radar_chart.png"), dpi=300, bbox_inches='tight')
plt.show()