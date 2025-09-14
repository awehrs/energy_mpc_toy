import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Flexible attention module that can handle different input/output dimensions."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        query_dim: Optional[int] = None,
        key_dim: Optional[int] = None,
        value_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Input/output dimensions (default to d_model if not specified)
        self.query_dim = query_dim if query_dim is not None else d_model
        self.key_dim = key_dim if key_dim is not None else d_model
        self.value_dim = value_dim if value_dim is not None else d_model
        self.output_dim = output_dim if output_dim is not None else d_model

        # Projections with flexible input dimensions
        self.q_proj = nn.Linear(self.query_dim, d_model, bias=False)
        self.k_proj = nn.Linear(self.key_dim, d_model, bias=False)
        self.v_proj = nn.Linear(self.value_dim, d_model, bias=False)

        # Output projection to desired output dimension
        self.out_proj = nn.Linear(d_model, self.output_dim)

        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, seq_len, query_dim]
            key: [batch, seq_len, key_dim] (None for self-attention)
            value: [batch, seq_len, value_dim] (None for self-attention)
            attn_mask: Optional attention mask

        Returns:
            output: [batch, seq_len, output_dim]
        """
        batch_size, q_len, _ = query.shape

        # For self-attention, use query as key and value
        if key is None:
            key = query
        if value is None:
            value = key

        k_len = key.shape[1]

        # Project to internal d_model dimension
        Q = self.q_proj(query)  # [batch, q_len, d_model]
        K = self.k_proj(key)  # [batch, k_len, d_model]
        V = self.v_proj(value)  # [batch, k_len, d_model]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, k_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, k_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Use PyTorch's Flash Attention (SDPA)
        attn_output = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )

        # Reshape back to [batch, q_len, d_model]
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, q_len, self.d_model)
        )

        # Project to output dimension
        return self.out_proj(attn_output)  # [batch, q_len, output_dim]


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 16,
        n_layers: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": Attention(
                            d_model=d_model, n_heads=n_heads, dropout=dropout
                        ),
                        "norm1": nn.LayerNorm(d_model),
                        "ffn": nn.Sequential(
                            nn.Linear(d_model, 4 * d_model),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(4 * d_model, d_model),
                            nn.Dropout(dropout),
                        ),
                        "norm2": nn.LayerNorm(d_model),
                    }
                )
                for _ in range(n_layers)
            ]
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: [batch, seq_len, d_model] - input embeddings
        Returns:
            latent: [batch, seq_len, d_model]
        """
        x = inputs

        for layer in self.layers:
            # Self-attention within latent
            attn_out = layer["self_attn"](x)
            x = layer["norm1"](x + attn_out)

            # Feed-forward
            ffn_out = layer["ffn"](x)
            x = layer["norm2"](x + ffn_out)

        return x


class DownSampler:
    """Cross Attention based downsampler."""
