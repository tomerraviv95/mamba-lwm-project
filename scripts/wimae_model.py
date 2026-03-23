import torch
import torch.nn as nn
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'WirelessContrastiveMaskedLearning'))
from contrawimae.models.modules.patching import Patcher
from contrawimae.models.modules.encoder import Encoder


class lwm_wimae(nn.Module):
    """
    WiMAE adapter that matches the lwm model interface for downstream benchmarking.

    Loads only the encoder (no decoder) from a WiMAE trainer checkpoint and exposes
    a forward() compatible with FineTuningWrapper.

    Input:  (B, 2, 32, 32) float tensor (real and imag stacked on dim 1)
    Output: (B, 128, 64) embeddings (128 patches, 64-dim features)
    """

    def __init__(self, checkpoint_path, device="cuda"):
        super().__init__()

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Support both trainer checkpoint (has "config") and model checkpoint (has "encoder_dim")
        if "config" in checkpoint:
            cfg = checkpoint["config"]
            # OmegaConf or dict - handle both
            if hasattr(cfg, "model"):
                patch_size = tuple(cfg.model.patch_size)
                encoder_dim = cfg.model.encoder_dim
                encoder_layers = cfg.model.encoder_layers
                encoder_nhead = cfg.model.encoder_nhead
            else:
                patch_size = tuple(cfg["model"]["patch_size"])
                encoder_dim = cfg["model"]["encoder_dim"]
                encoder_layers = cfg["model"]["encoder_layers"]
                encoder_nhead = cfg["model"]["encoder_nhead"]
        else:
            patch_size = checkpoint["patch_size"]
            encoder_dim = checkpoint["encoder_dim"]
            encoder_layers = checkpoint["encoder_layers"]
            encoder_nhead = checkpoint["encoder_nhead"]

        self.d_model = encoder_dim
        self.patch_size = patch_size
        patch_dim = patch_size[0] * patch_size[1]

        # Patcher is not an nn.Module, just a utility
        self.patcher = Patcher(patch_size)

        # Build encoder
        self.encoder = Encoder(
            input_dim=patch_dim,
            d_model=encoder_dim,
            nhead=encoder_nhead,
            num_layers=encoder_layers,
            mask_ratio=0.0,  # No masking for downstream
            device=torch.device(device),
        )

        # Load encoder weights from checkpoint
        model_state_dict = checkpoint["model_state_dict"]
        encoder_state_dict = {}
        for k, v in model_state_dict.items():
            if k.startswith("encoder."):
                encoder_state_dict[k[len("encoder."):]] = v

        self.encoder.load_state_dict(encoder_state_dict, strict=False)

    def forward(self, x, masked_pos=None, patch_dim=None):
        """
        Args:
            x: (B, 2, 32, 32) float tensor - real and imag stacked on dim 1
            masked_pos: unused, kept for interface compatibility
            patch_dim: unused, kept for interface compatibility

        Returns:
            (B, 128, 64) embeddings
        """
        # Reconstruct complex channel matrix
        complex_channels = torch.complex(x[:, 0], x[:, 1])  # (B, 32, 32)

        # Patch: returns (B, 2*P, L) = (B, 128, 16) for 32x32 with (16,1) patches
        patches = self.patcher(complex_channels)

        # Encode without masking: (B, 128, 64)
        embeddings = self.encoder(patches, apply_mask=False)

        return embeddings
