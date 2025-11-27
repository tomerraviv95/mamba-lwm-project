import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class ScaledDotProductAttention(nn.Module):
    """
    Computes scaled dot-product attention.

    Args:
        d_k (int): Dimensionality of the key vectors.
    """
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    Args:
        d_model (int): Total input/output dimension.
        n_heads (int): Number of attention heads.
        dropout (float): Dropout probability applied after attention.
    """
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, self.d_k * n_heads)
        self.W_K = nn.Linear(d_model, self.d_k * n_heads)
        self.W_V = nn.Linear(d_model, self.d_v * n_heads)
        self.linear = nn.Linear(n_heads * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_k)

    def forward(self, Q, K, V):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        context, attn = self.scaled_dot_attn(q_s, k_s, v_s)
        output = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(output)
        return residual + self.dropout(output), attn

class PoswiseFeedForwardNet(nn.Module):
    """
    Position-wise feed-forward network applied to each token independently.

    Args:
        d_model (int): Input and output dimensionality.
        d_ff (int): Hidden layer size in the feed-forward block.
        dropout (float): Dropout rate applied between layers.
    """
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    """
    Transformer encoder block composed of multi-head self-attention,
    feed-forward network, and layer normalization.

    Args:
        d_model (int): Embedding dimension.
        n_heads (int): Number of attention heads.
        d_ff (int): Hidden size of the feed-forward subnetwork.
        dropout (float): Dropout probability.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)

    def forward(self, enc_inputs):
        attn_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        attn_outputs = self.norm1(enc_inputs + attn_outputs)
        ff_outputs = self.pos_ffn(attn_outputs)
        enc_outputs = self.norm2(attn_outputs + ff_outputs)
        return enc_outputs, attn

class lwm(nn.Module):
    """
    Large Wireless Model (LWM): A Transformer-based encoder model for 
    extracting rich embeddings from wireless channel data.

    Args:
        element_length (int): Dimensionality of input tokens.
        d_model (int): Embedding dimension used throughout the network.
        n_layers (int): Number of Transformer encoder layers.
        max_len (int): Maximum number of tokens (sequence length).
        n_heads (int): Number of self-attention heads.
        dropout (float): Dropout probability used across the model.
    """
    def __init__(self, element_length=32, d_model=128, n_layers=12, max_len=513, n_heads=8, dropout=0.1):
        super().__init__()
        
        self.element_length = element_length
        self.d_model        = d_model
        self.n_layers       = n_layers
        self.max_len        = max_len
        self.n_heads        = n_heads
        self.dropout        = dropout
        
        self.embedding = Embedding(element_length, d_model, max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_model*4, dropout) for _ in range(n_layers)]
        )
        self.linear = nn.Linear(d_model, d_model)
        self.norm = LayerNormalization(d_model)

        embed_weight = self.embedding.proj.weight
        _, n_dim = embed_weight.size()
        self.decoder = nn.Linear(d_model, n_dim, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(n_dim))

    def forward(self, input_ids, masked_pos=None):
        """
        Forward pass of the LWM model.

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
        output = self.embedding(input_ids)

        for layer in self.layers:
            output, attn = layer(output)

        if masked_pos is not None:
            masked_pos = masked_pos.long()[:, :, None].expand(-1, -1, output.size(-1))
            h_masked = torch.gather(output, 1, masked_pos)
            h_masked = self.norm(F.relu(self.linear(h_masked)))
            logits_lm = self.decoder(h_masked) + self.decoder_bias
            return logits_lm, output
        else:
            return output