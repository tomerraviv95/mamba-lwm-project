"""Bidirectional Mamba SSM primitives, shared across LWM backbones.

Copied verbatim from ``scripts/mamba_model.py`` so both the DeepMIMO-channel and the
spectrogram backbones build on the same blocks. ``scripts/mamba_model.py`` keeps its own
copy for now; it can migrate to import from here in a later cleanup.
"""
import torch
import torch.nn as nn

# Import official Mamba SSM block
try:
    from mamba_ssm import Mamba as MambaSSM
except ImportError:
    raise ValueError("mamba_ssm not available. Install with: pip install mamba-ssm")


class MambaBlock(nn.Module):
    """
    Bidirectional Mamba block that processes sequences in both forward and backward directions.

    This enables BERT-like bidirectional context for masked language modeling:
    - Forward Mamba: captures left context
    - Backward Mamba: captures right context
    - Combined: each position sees full bidirectional context

    When bidirectional=True, each direction uses d_model//2 to keep total parameters similar.

    Args:
        d_model (int): Model dimension (output dimension)
        d_state (int): State dimension (N in the paper)
        d_conv (int): Convolution kernel size
        expand (int): Expansion factor for inner dimension
        dropout (float): Dropout rate
        bidirectional (bool): Whether to use bidirectional processing (default: True)
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1, bidirectional=True,
                 use_fast_path=True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout_p = dropout
        self.bidirectional = bidirectional
        # use_fast_path=False routes the SSM through nn.Module .forward / plain .weight access (the slow
        # path) so weight-parametrization LoRA on the SSM projections is exercised during downstream
        # finetuning. Pretraining leaves it True for the fused-kernel speed.

        if self.bidirectional:
            # Each direction gets half the dimension to maintain similar parameter count
            d_inner = d_model // 2

            # Project input to split dimensions for bidirectional processing
            self.input_proj = nn.Linear(d_model, d_inner * 2)

            # Forward Mamba (processes sequence left-to-right)
            self.mamba_forward = MambaSSM(
                d_model=d_inner,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                use_fast_path=use_fast_path,
            )

            # Backward Mamba (processes sequence right-to-left)
            self.mamba_backward = MambaSSM(
                d_model=d_inner,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                use_fast_path=use_fast_path,
            )

            # Output projection to combine forward and backward
            self.output_proj = nn.Linear(d_inner * 2, d_model)
        else:
            # Unidirectional: use full d_model
            self.mamba_forward = MambaSSM(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                use_fast_path=use_fast_path,
            )

        # Normalization and dropout
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.input_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (B, L, D) where B=batch, L=length, D=d_model
        Returns:
            output: (B, L, D)
        """
        residual = x
        x = self.input_norm(x)

        if self.bidirectional:
            # Project and split for bidirectional processing
            x_proj = self.input_proj(x)
            x_fwd, x_bwd = x_proj.chunk(2, dim=-1)  # Split into two halves

            # Forward pass: left-to-right
            output_fwd = self.mamba_forward(x_fwd)

            # Backward pass: right-to-left
            x_bwd_reversed = torch.flip(x_bwd, dims=[1])  # Reverse sequence dimension
            output_bwd = self.mamba_backward(x_bwd_reversed)
            output_bwd = torch.flip(output_bwd, dims=[1])  # Flip back to original order

            # Combine forward and backward: concatenate + project back to d_model
            output = torch.cat([output_fwd, output_bwd], dim=-1)
            output = self.output_proj(output)
        else:
            # Unidirectional: only forward pass
            output = self.mamba_forward(x)

        output = self.dropout(output)
        return self.norm(residual + output)


class MambaLayer(nn.Module):
    """
    Complete Mamba layer with Mamba block and feedforward network.

    Args:
        d_model (int): Model dimension
        d_state (int): SSM state dimension
        d_conv (int): Convolution kernel size
        expand (int): Expansion factor
        d_ff (int): Feedforward hidden dimension
        dropout (float): Dropout rate
        bidirectional (bool): Whether to use bidirectional Mamba
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, d_ff=512, dropout=0.1, bidirectional=True,
                 use_fast_path=True):
        super().__init__()
        self.mamba = MambaBlock(d_model, d_state, d_conv, expand, dropout, bidirectional,
                                use_fast_path=use_fast_path)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Mamba block
        x = self.mamba(x)

        # Feedforward with residual
        residual = x
        x = self.ffn(x)
        x = self.norm(residual + x)

        return x
