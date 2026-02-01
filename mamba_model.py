import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1, bidirectional=True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout_p = dropout
        self.bidirectional = bidirectional

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
            )

            # Backward Mamba (processes sequence right-to-left)
            self.mamba_backward = MambaSSM(
                d_model=d_inner,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
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
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, d_ff=512, dropout=0.1, bidirectional=True):
        super().__init__()
        self.mamba = MambaBlock(d_model, d_state, d_conv, expand, dropout, bidirectional)

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


class lwm_mamba(nn.Module):
    """
    Large Wireless Model (LWM) with Mamba architecture: A selective state space model
    for extracting rich embeddings from wireless channel data.

    Note: No positional encoding is used because Mamba's selective state-space mechanism
    inherently captures positional information through sequential processing.

    Supports bidirectional processing for masked language modeling tasks.

    **Multi-Resolution Support:**
    When patch_sizes is provided, the model uses resolution-specific embeddings W_emb(L)
    for each patch size L. This allows the same backbone to handle different tokenization
    granularities efficiently. Each W_emb(L) is a small learnable projection from
    patch_dimension to d_model.

    Args:
        element_length (int or list): Dimensionality of input tokens. Can be a single value
                                       or a list of values corresponding to patch_sizes.
        d_model (int): Embedding dimension used throughout the network.
        n_layers (int): Number of Mamba layers.
        max_len (int): Maximum number of tokens (sequence length).
        d_state (int): SSM state dimension.
        d_conv (int): Convolution kernel size.
        expand (int): Expansion factor for Mamba blocks.
        dropout (float): Dropout probability used across the model.
        bidirectional (bool): Whether to use bidirectional Mamba (default: True for MLM).
        patch_sizes (list, optional): List of patch sizes to support (e.g., [2, 4, 6, 8]).
                                      If provided, creates resolution-specific embeddings.
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
        dropout=0.1,
        bidirectional=True,
        patch_sizes=None
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.max_len = max_len
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.patch_sizes = patch_sizes

        # Multi-resolution support
        if patch_sizes is not None:
            # Create resolution-specific embeddings for each patch size
            # W_emb(L): patch_dim -> d_model
            self.resolution_embeddings = nn.ModuleDict()
            for patch_size in patch_sizes:
                patch_dim = patch_size * patch_size * 2  # patch_rows × patch_cols × 2 (real+imag)
                self.resolution_embeddings[str(patch_dim)] = nn.Linear(patch_dim, d_model)

            self.element_length = None  # Variable, depends on input
            print(f"Multi-resolution Mamba initialized with patch sizes: {patch_sizes}")
            print(f"Resolution-specific embeddings created for dimensions: {[p*p*2 for p in patch_sizes]}")
        else:
            # Single resolution (original behavior)
            self.element_length = element_length
            self.proj = nn.Linear(element_length, d_model)
            self.resolution_embeddings = None

        self.input_norm = nn.LayerNorm(d_model)

        # Stack of Mamba layers
        self.layers = nn.ModuleList([
            MambaLayer(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                d_ff=d_model * 4,
                dropout=dropout,
                bidirectional=bidirectional
            )
            for _ in range(n_layers)
        ])

        # Output projection for masked token prediction
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        # Decoder (resolution-specific for multi-resolution mode)
        if patch_sizes is not None:
            # Create resolution-specific decoders for each patch size
            # decoder(L): d_model -> patch_dim
            self.resolution_decoders = nn.ModuleDict()
            self.resolution_decoder_biases = nn.ParameterDict()
            for patch_size in patch_sizes:
                patch_dim = patch_size * patch_size * 2
                self.resolution_decoders[str(patch_dim)] = nn.Linear(d_model, patch_dim, bias=False)
                self.resolution_decoder_biases[str(patch_dim)] = nn.Parameter(torch.zeros(patch_dim))

            self.decoder = None  # Not used in multi-resolution mode
            self.decoder_bias = None
        else:
            # Single resolution decoder
            self.decoder = nn.Linear(d_model, element_length, bias=False)
            self.decoder_bias = nn.Parameter(torch.zeros(element_length))
            self.resolution_decoders = None
            self.resolution_decoder_biases = None

    def forward(self, input_ids, masked_pos=None, patch_dim=None):
        """
        Forward pass of the LWM Mamba model.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (B, T, element_length), where
                                      B is batch size, T is sequence length.
            masked_pos (torch.Tensor, optional): Indices of masked positions for patch prediction.
                                                 If provided, returns logits for these positions.
            patch_dim (int, optional): Patch dimension (element_length) for multi-resolution mode.
                                       Required when using resolution-specific embeddings.

        Returns:
            Tuple[torch.Tensor, torch.Tensor] if masked_pos is provided:
                - logits_lm: Predicted values for masked positions.
                - output: Full contextualized embeddings for all tokens.

            torch.Tensor if masked_pos is None:
                - output: Full contextualized embeddings of shape (B, T, d_model).
        """
        # Input projection (resolution-specific or single)
        if self.resolution_embeddings is not None:
            # Multi-resolution mode: use resolution-specific embedding
            if patch_dim is None:
                # Infer patch_dim from input shape
                patch_dim = input_ids.shape[-1]

            patch_dim_str = str(patch_dim)
            if patch_dim_str not in self.resolution_embeddings:
                raise ValueError(
                    f"Patch dimension {patch_dim} not supported. "
                    f"Available dimensions: {list(self.resolution_embeddings.keys())}"
                )

            # Use resolution-specific embedding W_emb(L)
            output = self.resolution_embeddings[patch_dim_str](input_ids.float())
        else:
            # Single resolution mode (original behavior)
            output = self.proj(input_ids.float())

        output = self.input_norm(output)

        # Pass through Mamba layers (shared backbone)
        for layer in self.layers:
            output = layer(output)

        # Masked token prediction if required
        if masked_pos is not None:
            masked_pos = masked_pos.long()[:, :, None].expand(-1, -1, output.size(-1))
            h_masked = torch.gather(output, 1, masked_pos)
            h_masked = self.norm(F.gelu(self.linear(h_masked)))

            # Use resolution-specific decoder if available
            if self.resolution_decoders is not None:
                if patch_dim is None:
                    patch_dim = input_ids.shape[-1]
                patch_dim_str = str(patch_dim)

                logits_lm = self.resolution_decoders[patch_dim_str](h_masked) + \
                           self.resolution_decoder_biases[patch_dim_str]
            else:
                # Single resolution decoder
                logits_lm = self.decoder(h_masked) + self.decoder_bias

            return logits_lm, output
        else:
            return output
