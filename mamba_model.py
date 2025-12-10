import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    Input embedding module with linear projection and positional encoding.

    Args:
        element_length (int): Length of each input element (e.g., patch size).
        d_model (int): Output embedding dimension.
        max_len (int): Maximum sequence length for positional embeddings.
    """
    def __init__(self, element_length, d_model, max_len=513):
        super().__init__()
        self.element_length = element_length
        self.d_model = d_model
        self.proj = nn.Linear(element_length, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.norm = LayerNormalization(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos_encodings = self.pos_embed(pos)
        tok_emb = self.proj(x.float())
        embedding = tok_emb + pos_encodings
        return self.norm(embedding)


class MambaBlock(nn.Module):
    """
    Simplified Mamba block using selective state space model principles.

    This implementation uses a selective mechanism to process sequences,
    similar to the Mamba architecture but simplified for this application.

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
        self.d_inner = int(expand * d_model)
        self.d_conv = d_conv

        # Input projection (produces x, z, B, C, dt for selective SSM)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        # Initialize with smaller std for stability
        nn.init.normal_(self.in_proj.weight, std=0.02)

        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        # Initialize conv weights
        nn.init.normal_(self.conv1d.weight, std=0.02)

        # SSM parameters projection
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)  # For B and C
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)  # For delta (time step)

        # State space parameters
        # A is initialized to be stable (negative real parts)
        # Use a range that ensures stability: A should be negative
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # Will be negative after exp
        self.D = nn.Parameter(torch.ones(self.d_inner))  # Skip connection parameter

        # Initialize projections with smaller values for stability
        nn.init.normal_(self.dt_proj.weight, std=0.02)
        nn.init.normal_(self.x_proj.weight, std=0.02)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        nn.init.normal_(self.out_proj.weight, std=0.02)

        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)
        self.input_norm = LayerNormalization(d_model)

    def forward(self, x):
        """
        Args:
            x: (B, L, D) where B=batch, L=length, D=d_model
        Returns:
            output: (B, L, D)
        """
        batch, seqlen, dim = x.shape
        residual = x

        # Normalize input first for stability
        x = self.input_norm(x)

        # Input projection: split into x and z (gating)
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)

        # Apply 1D convolution for local context
        # Rearrange for conv1d: (B, d_inner, L)
        x_conv = rearrange(x_in, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[..., :seqlen]  # Trim to original length
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        x_conv = F.silu(x_conv)

        # Selective SSM
        # Project to get B and C matrices (input-dependent)
        x_proj_out = self.x_proj(x_conv)  # (B, L, 2*d_state)
        B, C = x_proj_out.chunk(2, dim=-1)  # Each (B, L, d_state)

        # Delta (time step) - also input dependent
        # Use softplus with clipping to avoid very large values
        delta = self.dt_proj(x_conv)  # (B, L, d_inner)
        delta = F.softplus(delta)
        # Clip delta to prevent numerical instability
        delta = torch.clamp(delta, min=1e-4, max=10.0)  # (B, L, d_inner)

        # Get A matrix (make it negative for stability)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Selective SSM scan (simplified sequential processing)
        y = self.selective_scan(x_conv, delta, A, B, C, self.D)

        # Gating mechanism
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)
        output = self.dropout(output)

        # Residual connection
        return self.norm(residual + output)

    def selective_scan(self, x, delta, A, B, C, D):
        """
        Numerically stable selective scan operation.

        Args:
            x: (B, L, d_inner)
            delta: (B, L, d_inner) - time step (already clamped)
            A: (d_inner, d_state) - state transition (negative values)
            B: (B, L, d_state) - input matrix
            C: (B, L, d_state) - output matrix
            D: (d_inner,) - skip connection
        Returns:
            y: (B, L, d_inner)
        """
        batch, seqlen, d_inner = x.shape
        d_state = A.shape[1]

        # Discretize A using delta
        # deltaA = exp(delta * A) where A is negative (d_inner, d_state)
        # delta is (B, L, d_inner) -> expand for broadcasting
        # Clamp the exponent to prevent overflow
        delta_A_exp = delta.unsqueeze(-1) * A  # (B, L, d_inner, d_state)
        delta_A_exp = torch.clamp(delta_A_exp, min=-10.0, max=0.0)  # A is negative, so result is negative
        deltaA = torch.exp(delta_A_exp)  # (B, L, d_inner, d_state)

        # deltaB * x
        # B is (B, L, d_state), x is (B, L, d_inner)
        deltaB_x = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)  # (B, L, d_inner, d_state)

        # Sequential scan through time with gradient checkpointing for stability
        # Initialize state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seqlen):
            # Update state: h = deltaA * h + deltaB_x
            h = deltaA[:, t] * h + deltaB_x[:, t]

            # Periodically renormalize to prevent explosion (every 16 steps)
            if t % 16 == 15:
                h_norm = torch.norm(h, dim=-1, keepdim=True)
                h_norm = torch.clamp(h_norm, min=1e-6)
                # Only normalize if values are getting large
                scale = torch.where(h_norm > 10.0, 10.0 / h_norm, torch.ones_like(h_norm))
                h = h * scale

            # Output: y = C * h + D * x
            y_t = torch.sum(C[:, t].unsqueeze(1) * h, dim=-1) + D * x[:, t]  # (B, d_inner)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)
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
