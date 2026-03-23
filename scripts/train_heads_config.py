import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import patch_reconstructor
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'WirelessContrastiveMaskedLearning'))
from contrawimae.models.modules.patching import InversePatcher

# Define TaskHead for each task
class LosNlosClassificationHead(nn.Module):
    """
    Task head for LoS/NLoS classification.

    Takes flattened patch embeddings as input and outputs class logits for binary classification.

    Args:
        input_dim (tuple): (n_patches, d_model) — number of patches and feature dimension.
    """
    def __init__(self, input_dim):
        super().__init__()
        n_patches, d_model = input_dim
        flattened_dim = n_patches * d_model 
        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, 8),  
            nn.BatchNorm1d(8),       
            nn.ReLU(),                 
            nn.Dropout(0.1),           
            nn.Linear(8, 2), 
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  
        x = self.classifier(x)  
        return x
    
class BeamPredictionHead(nn.Module):
    """
    Task head for mmWave beam index prediction.

    Processes flattened patch embeddings and outputs logits over 64 possible beam indices.

    Args:
        input_dim (tuple): (n_patches, d_model) — number of patches and feature dimension.
    """
    def __init__(self, input_dim):
        super().__init__()
        n_patches, d_model = input_dim
        flattened_dim = n_patches * d_model 
        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, 256),  
            nn.BatchNorm1d(256),       
            nn.ReLU(),                 
            nn.Dropout(0.1),           
            nn.Linear(256, 128),       
            nn.BatchNorm1d(128),       
            nn.ReLU(),                 
            nn.Linear(128, 64)
        )
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  
        x = self.classifier(x)  
        return x

class ChannelInterpolationHead(nn.Module):
    """
    Task head for reconstructing missing channel values from patch embeddings.

    Applies a linear layer to each patch and reconstructs the full channel using patch_reconstructor.

    Args:
        input_dim (tuple): (n_patches, d_model).
        output_dim (tuple): (target_channels, n_rows, n_cols) — shape of the output channel matrix.
        patch_size (int): Spatial patch size (e.g., 2, 4, 6, 8). Each patch has patch_size*patch_size*2 values.
    """
    def __init__(self, input_dim, output_dim, patch_size=4):
        super().__init__()
        n_patches, d_model = input_dim
        target_channels, self.n_rows, self.n_cols = output_dim
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size * 2
        self.fcn = nn.Sequential(
            nn.Linear(d_model, self.patch_dim)
        )

    def forward(self, x):
        batch_size, n_patches, d_model = x.size()
        x = x.reshape(batch_size * n_patches, d_model)
        x = self.fcn(x)
        x = x.reshape(batch_size, n_patches, self.patch_dim)
        x = patch_reconstructor(x, self.n_rows, self.n_cols,
                                patch_rows=self.patch_size, patch_cols=self.patch_size)
        return x

class ChannelEstimationHead(nn.Module):
    """
    Task head for full channel estimation from embeddings.

    Similar to interpolation but typically used for denoising or noisy reconstruction.

    Args:
        input_dim (tuple): (n_patches, d_model).
        output_dim (tuple): (target_channels, n_rows, n_cols) — shape of the target full-resolution channel.
        patch_size (int): Spatial patch size (e.g., 2, 4, 6, 8). Each patch has patch_size*patch_size*2 values.
    """
    def __init__(self, input_dim, output_dim, patch_size=4):
        super().__init__()
        n_patches, d_model = input_dim
        target_channels, self.n_rows, self.n_cols = output_dim
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size * 2
        self.fcn = nn.Sequential(
            nn.Linear(d_model, self.patch_dim)
        )

    def forward(self, x):
        batch_size, n_patches, d_model = x.size()
        x = x.reshape(batch_size * n_patches, d_model)
        x = self.fcn(x)
        x = x.reshape(batch_size, n_patches, self.patch_dim)
        x = patch_reconstructor(x, self.n_rows, self.n_cols,
                                patch_rows=self.patch_size, patch_cols=self.patch_size)
        return x

class ChannelChartingHead(nn.Module):
    """
    Task head for 2D channel charting (e.g., learning spatial topology).

    Reduces the flattened embeddings into 2D coordinates.

    Args:
        input_dim (tuple): (n_patches, d_model).
    """
    def __init__(self, input_dim):
        super().__init__()
        n_patches, d_model = input_dim
        flattened_dim = n_patches * d_model 
        self.fcn = nn.Sequential(
            nn.Linear(flattened_dim, 64),      
            nn.ReLU(),                       
            nn.Linear(64, 32),        
            nn.ReLU(),                      
            nn.Linear(32, 2)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fcn(x)  
        return x

class WiMAEChannelReconstructionHead(nn.Module):
    """
    Task head for channel reconstruction (interpolation/estimation) with WiMAE embeddings.

    Projects per-patch embeddings back to WiMAE patch space, uses InversePatcher to
    reconstruct a 32x32 complex channel, then resizes to target resolution.

    Args:
        input_dim (tuple): (n_patches, d_model) — (128, 64) for WiMAE.
        output_dim (tuple): (target_channels, n_rows, n_cols) — target output shape.
    """
    def __init__(self, input_dim, output_dim, patch_size=None):
        super().__init__()
        n_patches, d_model = input_dim
        self.target_channels, self.n_rows, self.n_cols = output_dim
        wimae_patch_dim = 16  # WiMAE patch dim: 16x1 = 16
        self.fc = nn.Linear(d_model, wimae_patch_dim)
        self.inverse_patcher = InversePatcher((32, 32), (16, 1))

    def forward(self, x):
        batch_size, n_patches, d_model = x.size()
        # Project embeddings to patch space: (B, 128, 16)
        x = x.reshape(batch_size * n_patches, d_model)
        x = self.fc(x)
        x = x.reshape(batch_size, n_patches, -1)
        # InversePatcher expects (B, 2*P, L) = (B, 128, 16)
        complex_channels = self.inverse_patcher(x)  # (B, 32, 32) complex
        # Stack real/imag: (B, 2, 32, 32)
        out = torch.stack([complex_channels.real, complex_channels.imag], dim=1)
        # Resize to target resolution if needed
        if self.n_rows != 32 or self.n_cols != 32:
            out = F.interpolate(out, size=(self.n_rows, self.n_cols),
                                mode="bilinear", align_corners=False)
        return out


# training_configs is a list of dictionaries, each specifying the setup for one downstream task.
# Each entry includes:
# - task: Name of the task.
# - optimizer_config: Learning rate for the optimizer.
# - scheduler: Step size and decay rate for the learning rate scheduler.
# - epochs: Total number of training epochs.
# - batch_size: Number of samples per batch.
# - loss_function: Loss type ("CrossEntropyLoss" or "MSELoss").
# - seed: Random seed for reproducibility.
# - fine_tune_layers: Specifies which parts of the LWM model to fine-tune:
#     • "full" means all layers are trainable
#     • A list like ["layers.10", "layers.11"] specifies partial fine-tuning
# - input_type: Type of embedding input used from LWM:
#     • "cls_emb", "channel_emb", "mean_pooled", etc.
# - selected_tokens: Specific tokens to select for input, used in some tasks.

training_configs = [
    {  # Task 1
        "task": "LosNlosClassification",
        "optimizer_config": {"lr": 1e-3},
        "scheduler": {"step_size": 20, "gamma": 0.5},
        "epochs": 200,
        "batch_size": 128,
        "seed": 42,
        "fine_tune_layers": ["layers.9", "layers.10", "layers.11"],
        "input_type": "cls_emb",
        "selected_tokens": None
    },
    {  # Task 2
        "task": "BeamPrediction",
        "optimizer_config": {"lr": 1e-3},
        "scheduler": {"step_size": 20, "gamma": 0.8},
        "epochs": 70,
        "batch_size": 128,
        "seed": 42,
        "fine_tune_layers": "full",
        "input_type": "mean_pooled",
        "selected_tokens": None
    },
    {  # Task 3
        "task": "ChannelInterpolation",
        "optimizer_config": {"lr": 1e-2},
        "scheduler": {"step_size": 25, "gamma": 0.2},
        "epochs": 100,
        "batch_size": 128,
        "seed": 42,
        "fine_tune_layers": ["layers.10", "layers.11"],
        "input_type": "channel_emb",
        "selected_tokens": None
    },
    {  # Task 4
        "task": "ChannelEstimation",
        "optimizer_config": {"lr": 1e-2},
        "scheduler": {"step_size": 50, "gamma": 0.3},
        "epochs": 200,
        "batch_size": 128,
        "seed": 42,
        "fine_tune_layers": ["layers.9", "layers.10", "layers.11"],
        "input_type": "channel_emb",
        "selected_tokens": None
    },
    {  # Task 5
        "task": "ChannelCharting",
        "optimizer_config": {"lr": 1e-3},
        "scheduler": {"step_size": 40, "gamma": 0.6},
        "epochs": 150,
        "batch_size": 128,
        "seed": 42,
        "fine_tune_layers": ["layers.10", "layers.11"],
        "input_type": "mean_pooled",
        "selected_tokens": None
    }
]
