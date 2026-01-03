from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Int, Int32, Float
from transformers.models.qwen2.modeling_qwen2 import rotate_half


try:
    from flash_attn import (
        flash_attn_kvpacked_func,
        flash_attn_qkvpacked_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
    )
except ImportError:
    flash_attn_varlen_qkvpacked_func, flash_attn_varlen_kvpacked_func = None, None
    flash_attn_qkvpacked_func, flash_attn_kvpacked_func = None, None
    flash_attn_with_kvcache = None


class FlashSelfAttention(nn.Module):

    def __init__(
        self,
        causal: bool = False,
        softmax_scale: Optional[float] = None,
        attention_dropout: float = 0.0,
        alibi_slopes: Optional[float] = None,
        window_size: Tuple[int] = (-1, -1),
        deterministic: bool = False,
    ):
        super().__init__()
        assert (
            flash_attn_varlen_qkvpacked_func is not None
        ), "FlashAttention is not installed"
        assert flash_attn_qkvpacked_func is not None, "FlashAttention is not installed"
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)
        self.window_size = window_size
        self.deterministic = deterministic

    def forward(
        self,
        qkv: Float[torch.Tensor, "batch seq_len 3 n_heads dim"],
        causal: Optional[bool] = None,
        max_seq_len: Optional[int] = None,
        cu_seq_lens: Optional[Int32[torch.Tensor, "batch+1"]] = None,
    ) -> Float[torch.Tensor, "batch q_len n_heads dim"]:

        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda
        causal = self.causal if causal is None else causal
        unpadded = cu_seq_lens is not None
        if self.alibi_slopes is not None:
            self.alibi_slopes = self.alibi_slopes.to(torch.float32)
        if unpadded:
            assert cu_seq_lens.dtype == torch.int32
            assert max_seq_len is not None
            assert isinstance(max_seq_len, int)
            return flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seq_lens,
                max_seq_len,
                self.drop.p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=causal,
                alibi_slopes=self.alibi_slopes,
                window_size=self.window_size,
                deterministic=self.deterministic,
            )
        else:
            return flash_attn_qkvpacked_func(
                qkv,
                self.drop.p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=causal,
                alibi_slopes=self.alibi_slopes,
                window_size=self.window_size,
                deterministic=self.deterministic,
            )


class FlashCrossAttention(nn.Module):

    def __init__(
        self,
        causal: bool = False,
        softmax_scale: Optional[float] = None,
        attention_dropout: float = 0.0,
        alibi_slopes: Optional[float] = None,
        window_size: Tuple[int] = (-1, -1),
        deterministic: bool = False,
    ):
        super().__init__()
        assert (
            flash_attn_varlen_kvpacked_func is not None
        ), "FlashAttention is not installed"
        assert flash_attn_kvpacked_func is not None, "FlashAttention is not installed"
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)
        self.window_size = window_size
        self.deterministic = deterministic

    def forward(
        self,
        q: Float[torch.Tensor, "batch q_len n_heads dim"],
        kv: Float[torch.Tensor, "batch kv_len 2 n_heads dim"],
        causal: Optional[bool] = None,
        max_seq_len_q: Optional[int] = None,
        max_seq_len_k: Optional[int] = None,
        cu_seq_lens_q: Optional[Int32[torch.Tensor, "batch+1"]] = None,
        cu_seq_lens_k: Optional[Int32[torch.Tensor, "batch+1"]] = None,
    ) -> Float[torch.Tensor, "batch q_len n_heads dim"]:

        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and kv.is_cuda

        causal = self.causal if causal is None else causal
        unpadded = cu_seq_lens_q is not None

        if self.alibi_slopes is not None:
            self.alibi_slopes = self.alibi_slopes.to(torch.float32)

        if unpadded:
            assert cu_seq_lens_q.dtype == torch.int32
            assert max_seq_len_q is not None
            assert isinstance(max_seq_len_q, int)
            assert cu_seq_lens_k is not None
            assert cu_seq_lens_k.dtype == torch.int32
            assert max_seq_len_k is not None
            assert isinstance(max_seq_len_k, int)
            return flash_attn_varlen_kvpacked_func(
                q,
                kv,
                cu_seq_lens_q,
                cu_seq_lens_k,
                max_seq_len_q,
                max_seq_len_k,
                self.drop.p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=causal,
                alibi_slopes=self.alibi_slopes,
                window_size=self.window_size,
                deterministic=self.deterministic,
            )
        else:
            batch_size, seqlen_q = q.shape[0], q.shape[1]
            seqlen_k = kv.shape[1]
            assert kv.shape[0] == batch_size and kv.shape[4] == q.shape[3]
            return flash_attn_kvpacked_func(
                q,
                kv,
                self.drop.p if self.training else 0.0,
                causal=causal,
                softmax_scale=self.softmax_scale,
                alibi_slopes=self.alibi_slopes,
                window_size=self.window_size,
                deterministic=self.deterministic,
            )


class Attention(nn.Module):
    """Flexible attention module that can handle different input/output dimensions."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        is_causal: bool = False,
        is_cross_attention: bool = False,
        use_flash_attention: bool = True,
        query_dim: Optional[int] = None,
        key_dim: Optional[int] = None,
        value_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.is_causal = is_causal
        self.is_cross_attention = is_cross_attention
        self.use_flash_attention = use_flash_attention

        if use_flash_attention:
            if is_cross_attention:
                self.attn = FlashCrossAttention(
                    causal=False,
                    attention_dropout=dropout,
                )
            else:
                self.attn = FlashSelfAttention(
                    causal=is_causal,
                    attention_dropout=dropout,
                )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

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
        self.out_proj = nn.Linear(d_model, self.output_dim, bias=False)

        self.dropout = dropout

    def forward(
        self,
        query: Float[torch.Tensor, "batch q_len q_dim"],
        key: Optional[Float[torch.Tensor, "batch kv_len k_dim"]] = None,
        position_emb: Optional[torch.Tensor] = None,
        max_seq_len_q: Optional[int] = None,
        max_seq_len_k: Optional[int] = None,
        cu_seq_lens_q: Optional[Int32[torch.Tensor, "batch+1"]] = None,
        cu_seq_lens_k: Optional[Int32[torch.Tensor, "batch+1"]] = None,
        attn_mask: Optional[Int[torch.Tensor, "batch seq_len"]] = None,
    ) -> Float[torch.Tensor, "batch q_len output_dim"]:

        is_batched = len(query.shape) == 3

        if not is_batched:
            assert (
                cu_seq_lens_q is not None
            ), "cu_seq_lens_q required for packed sequences"
            assert (
                max_seq_len_q is not None
            ), "max_seq_len_q required for packed sequences"

        if key is None:
            key = query

        if is_batched:
            batch_size, q_len, _ = query.shape
            k_len = key.shape[1]
        else:
            total_q_len = query.shape[0]
            total_k_len = key.shape[0]

        # Project
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(key)

        # Reshape for multi-head attention
        if is_batched:
            Q = Q.view(batch_size, q_len, self.n_heads, self.head_dim)
            K = K.view(batch_size, k_len, self.n_heads, self.head_dim)
            V = V.view(batch_size, k_len, self.n_heads, self.head_dim)
        else:
            Q = Q.view(total_q_len, self.n_heads, self.head_dim)
            K = K.view(total_k_len, self.n_heads, self.head_dim)
            V = V.view(total_k_len, self.n_heads, self.head_dim)

        # Apply RoPE (for cross attention context only)
        if position_emb is not None:
            cos, sin = position_emb

            # Remove dummy batch dim for packed case
            if not is_batched and cos.dim() == 3:
                cos = cos.squeeze(0)
                sin = sin.squeeze(0)

            # Determine unsqueeze dimension
            unsqueeze_dim = 2 if is_batched else 1

            # Apply RoPE manually to K only
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)
            K = (K * cos) + (rotate_half(K) * sin)

        # Flash Attention
        if self.use_flash_attention:

            if self.is_cross_attention:
                if is_batched:
                    kv = torch.stack([K, V], dim=2)
                else:
                    kv = torch.stack([K, V], dim=1)

                attn_output = self.attn(
                    q=Q,
                    kv=kv,
                    max_seq_len_q=max_seq_len_q,
                    max_seq_len_k=max_seq_len_k,
                    cu_seq_lens_q=cu_seq_lens_q,
                    cu_seq_lens_k=cu_seq_lens_k,
                )
            else:
                if is_batched:
                    qkv = torch.stack([Q, K, V], dim=2)
                else:
                    qkv = torch.stack([Q, K, V], dim=1)

                attn_output = self.attn(
                    qkv=qkv,
                    max_seq_len=max_seq_len_q,
                    cu_seq_lens=cu_seq_lens_q,
                )

            # Reshape output
            if is_batched:
                attn_output = attn_output.reshape(batch_size, q_len, -1)
            else:
                attn_output = attn_output.reshape(total_q_len, -1)

        # SDPA fallback
        else:
            assert is_batched, "SDPA fallback only supports batched sequences"
            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1).float()
                attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf"))
                attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)

            attn_output = F.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=attn_mask,
                is_causal=self.is_causal,
                dropout_p=self.dropout if self.training else 0.0,
            )
            attn_output = (
                attn_output.transpose(1, 2)
                .contiguous()
                .view(batch_size, q_len, self.d_model)
            )

        return self.out_proj(attn_output)


class MLP(nn.Module):
    """MLP with SwiGLU activation function."""

    def __init__(self, d_model: int):
        super().__init__()

        hidden_dim = int(8 * d_model / 3)

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(x))
        data = self.w2(x)

        return self.w3(gate * data)


class TransformerBlock(nn.Module):
    """Transformer block with pre-RMSNorm"""

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
                        "norm1": nn.RMSNorm(d_model),
                        "self_attn": Attention(
                            d_model=d_model, n_heads=n_heads, dropout=dropout
                        ),
                        "norm2": nn.RMSNorm(d_model),
                        "ffn": MLP(d_model),
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
        elif isinstance(module, nn.RMSNorm):
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        q: Float[torch.Tensor, "batch q_len n_heads dim"],
        kv: Optional[Float[torch.Tensor, "batch kv_len 2 n_heads dim"]] = None,
        position_emb: Optional[torch.Tensor] = None,
        max_seq_len_q: Optional[int] = None,
        max_seq_len_k: Optional[int] = None,
        cu_seq_lens_q: Optional[Int32[torch.Tensor, "batch+1"]] = None,
        cu_seq_lens_k: Optional[Int32[torch.Tensor, "batch+1"]] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        hidden_states = q
        context = kv
        had_cross_attention = context is not None

        for layer in self.layers:

            residual = hidden_states

            if context is None:
                # Self attention
                hidden_states = layer["norm1"](hidden_states)
            else:
                # Cross attention
                context = layer["norm1"](context)

            hidden_states = layer["self_attn"](
                query=hidden_states,
                key=context,
                position_emb=position_emb,
                max_seq_len_q=max_seq_len_q,
                max_seq_len_k=max_seq_len_k,
                cu_seq_lens_q=cu_seq_lens_q,
                cu_seq_lens_k=cu_seq_lens_k,
                attn_mask=attn_mask,
            )

            # After cross-attention, disable position_emb for remaining layers
            if had_cross_attention:
                context = None
                position_emb = None

            # Residual stream.
            hidden_states = hidden_states + residual

            residual = hidden_states

            # Feed-forward
            hidden_states = layer["norm2"](hidden_states)
            hidden_states = layer["ffn"](hidden_states)

            # Residual stream.
            hidden_states = hidden_states + residual

        return hidden_states
