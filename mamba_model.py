import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import official Mamba SSM block
try:
    from mamba_ssm import Mamba as MambaSSM
    MAMBA_SSM_AVAILABLE = True
except ImportError:
    MAMBA_SSM_AVAILABLE = False
    MambaSSM = None
    print("Warning: mamba_ssm not available. Install with: pip install mamba-ssm")

# Use einops if available, otherwise use manual rearrange
try:
    from einops import rearrange
except ImportError:
    def rearrange(tensor, pattern, **axes_lengths):
        """Simplified rearrange for 'b l d -> b d l' and 'b d l -> b l d'."""
        if pattern == 'b l d -> b d l':
            return tensor.transpose(1, 2)
        elif pattern == 'b d l -> b l d':
            return tensor.transpose(1, 2)
        else:
            raise NotImplementedError(f"Pattern {pattern} not implemented in fallback rearrange")

class LayerNormalization(nn.Module):
    """
    Custom Layer Normalization module with learnable scale and bias.

    Args:
        d_model (int): Dimensionality of the input embeddings.
        eps (float): A small constant added to variance to avoid division by zero.
    """
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class Embedding(nn.Module):
    """
    Input embedding module with linear projection (no positional encoding for Mamba).

    Note: Mamba doesn't need positional encoding because its selective state-space
    mechanism inherently captures positional information through sequential processing.

    Args:
        element_length (int): Length of each input element (e.g., patch size).
        d_model (int): Output embedding dimension.
        max_len (int): Maximum sequence length (kept for API compatibility but not used).
    """
    def __init__(self, element_length, d_model, max_len=513):
        super().__init__()
        self.element_length = element_length
        self.d_model = d_model
        self.proj = nn.Linear(element_length, d_model)
        self.norm = LayerNormalization(d_model)

    def forward(self, x):
        tok_emb = self.proj(x.float())
        return self.norm(tok_emb)


class MambaBlock(nn.Module):
    """
    Mamba block wrapper that uses the official mamba-ssm implementation when available,
    otherwise falls back to a custom implementation.

    This wrapper maintains compatibility with the rest of the model while using
    optimized fused kernels from the mamba-ssm package.

    Args:
        d_model (int): Model dimension
        d_state (int): State dimension (N in the paper)
        d_conv (int): Convolution kernel size
        expand (int): Expansion factor for inner dimension
        dropout (float): Dropout rate
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout_p = dropout

        if MAMBA_SSM_AVAILABLE:
            # Use official mamba-ssm implementation with fused kernels
            self.mamba = MambaSSM(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.use_official = True
        else:
            # Fallback to custom implementation
            self.use_official = False
            self._init_custom_implementation(d_model, d_state, d_conv, expand)

        # Normalization and dropout (used in both cases)
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)
        self.input_norm = LayerNormalization(d_model)

    def _init_custom_implementation(self, d_model, d_state, d_conv, expand):
        """Initialize custom Mamba implementation when official package is not available."""
        self.d_inner = int(expand * d_model)

        # Input projection (produces x, z, B, C, dt for selective SSM)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        nn.init.normal_(self.in_proj.weight, std=0.02)

        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        nn.init.normal_(self.conv1d.weight, std=0.02)

        # SSM parameters projection
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)

        # State space parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Initialize projections
        nn.init.normal_(self.dt_proj.weight, std=0.02)
        nn.init.normal_(self.x_proj.weight, std=0.02)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def forward(self, x):
        """
        Args:
            x: (B, L, D) where B=batch, L=length, D=d_model
        Returns:
            output: (B, L, D)
        """
        residual = x
        x = self.input_norm(x)

        if self.use_official:
            # Use official mamba-ssm implementation
            output = self.mamba(x)
        else:
            # Use custom implementation
            output = self._forward_custom(x)

        output = self.dropout(output)
        return self.norm(residual + output)

    def _forward_custom(self, x):
        """Custom Mamba forward pass for fallback."""
        seqlen = x.size(1)

        # Input projection: split into x and z (gating)
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        # Apply 1D convolution for local context
        x_conv = rearrange(x_in, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[..., :seqlen]
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        x_conv = F.silu(x_conv)

        # Selective SSM
        x_proj_out = self.x_proj(x_conv)
        B, C = x_proj_out.chunk(2, dim=-1)

        # Delta (time step)
        delta = self.dt_proj(x_conv)
        delta = F.softplus(delta)
        delta = torch.clamp(delta, min=1e-4, max=10.0)

        # Get A matrix
        A = -torch.exp(self.A_log)

        # Selective SSM scan
        y = self._selective_scan(x_conv, delta, A, B, C, self.D)

        # Gating mechanism
        y = y * F.silu(z)

        # Output projection
        return self.out_proj(y)

    def _selective_scan(self, x, delta, A, B, C, D):
        """Numerically stable selective scan operation."""
        batch, seqlen, d_inner = x.shape
        d_state = A.shape[1]

        # Discretize A using delta
        delta_A_exp = delta.unsqueeze(-1) * A
        delta_A_exp = torch.clamp(delta_A_exp, min=-10.0, max=0.0)
        deltaA = torch.exp(delta_A_exp)

        # deltaB * x
        deltaB_x = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)

        # Initialize state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seqlen):
            # Update state
            h = deltaA[:, t] * h + deltaB_x[:, t]

            # Periodically renormalize
            if t % 16 == 15:
                h_norm = torch.norm(h, dim=-1, keepdim=True)
                h_norm = torch.clamp(h_norm, min=1e-6)
                scale = torch.where(h_norm > 10.0, 10.0 / h_norm, torch.ones_like(h_norm))
                h = h * scale

            # Output
            y_t = torch.sum(C[:, t].unsqueeze(1) * h, dim=-1) + D * x[:, t]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return y


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
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, d_ff=512, dropout=0.1):
        super().__init__()
        self.mamba = MambaBlock(d_model, d_state, d_conv, expand, dropout)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = LayerNormalization(d_model)

    def forward(self, x):
        # Mamba block
        x = self.mamba(x)

        # Feedforward with residual
        residual = x
        x = self.ffn(x)
        x = self.norm(residual + x)

        return x


class lwm_mamba(nn.Module):
    """
    Large Wireless Model (LWM) with Mamba architecture: A selective state space model
    for extracting rich embeddings from wireless channel data.

    Args:
        element_length (int): Dimensionality of input tokens.
        d_model (int): Embedding dimension used throughout the network.
        n_layers (int): Number of Mamba layers.
        max_len (int): Maximum number of tokens (sequence length).
        d_state (int): SSM state dimension.
        d_conv (int): Convolution kernel size.
        expand (int): Expansion factor for Mamba blocks.
        dropout (float): Dropout probability used across the model.
    """
    def __init__(
        self,
        element_length=32,
        d_model=128,
        n_layers=12,
        max_len=513,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1
    ):
        super().__init__()

        self.element_length = element_length
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_len = max_len
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout = dropout

        # Embedding layer
        self.embedding = Embedding(element_length, d_model, max_len)

        # Stack of Mamba layers
        self.layers = nn.ModuleList([
            MambaLayer(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                d_ff=d_model * 4,  # Match transformer's d_ff
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Output projection for masked token prediction
        self.linear = nn.Linear(d_model, d_model)
        self.norm = LayerNormalization(d_model)

        # Decoder (tied with embedding weights conceptually)
        embed_weight = self.embedding.proj.weight
        _, n_dim = embed_weight.size()
        self.decoder = nn.Linear(d_model, n_dim, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(n_dim))

    def forward(self, input_ids, masked_pos=None):
        """
        Forward pass of the LWM Mamba model.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (B, T, element_length), where
                                      B is batch size, T is sequence length.
            masked_pos (torch.Tensor, optional): Indices of masked positions for patch prediction.
                                                 If provided, returns logits for these positions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor] if masked_pos is provided:
                - logits_lm: Predicted values for masked positions.
                - output: Full contextualized embeddings for all tokens.

            torch.Tensor if masked_pos is None:
                - output: Full contextualized embeddings of shape (B, T, d_model).
        """
        # Embedding
        output = self.embedding(input_ids)

        # Pass through Mamba layers
        for layer in self.layers:
            output = layer(output)

        # Masked token prediction if required
        if masked_pos is not None:
            masked_pos = masked_pos.long()[:, :, None].expand(-1, -1, output.size(-1))
            h_masked = torch.gather(output, 1, masked_pos)
            h_masked = self.norm(F.gelu(self.linear(h_masked)))
            logits_lm = self.decoder(h_masked) + self.decoder_bias
            return logits_lm, output
        else:
            return output
