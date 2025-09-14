from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.cache_utils import DynamicCache

from models.utils import Attention


class CrossAttentionAdapter(nn.Module):
    """
    Adapter to bridge custom latent representations to KV cache prefixes.

    Works only with decoder only LLMs.

    WARNING: will not work, as is, with ROPE-based models.
    """

    def __init__(
        self,
        latent_dim: int,
        decoder_hidden_dim: int,
        decoder_n_heads: int,
        decoder_n_layers: int,
        n_heads: int = 8,
        num_prefix_tokens: Optional[int] = 16,
        adapter_dim: Optional[int] = None,
        dropout: int = 0.1,
    ):
        super().__init__()
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_n_heads = decoder_n_heads
        self.d_head = decoder_hidden_dim // decoder_n_heads
        self.num_prefix_tokens = num_prefix_tokens
        self.num_layers = decoder_n_layers

        if adapter_dim is None:
            adapter_dim = decoder_hidden_dim // 4

        # Learnable prefix queries for downsampling
        self.prefix_queries = nn.Parameter(
            torch.randn(num_prefix_tokens, latent_dim) * 0.02
        )

        # Cross-attention to select from latent
        self.cross_attention = Attention(
            d_model=latent_dim,
            n_heads=n_heads,
            query_dim=latent_dim,
            key_dim=latent_dim,
            value_dim=latent_dim,
        )

        # Projection + adapter in decoder space
        self.latent_proj = nn.Linear(latent_dim, decoder_hidden_dim)

        self.adapter = nn.Sequential(
            nn.Linear(decoder_hidden_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, decoder_hidden_dim),
        )

        self.norm = nn.LayerNorm(decoder_hidden_dim)

        # Separate k/v projections
        self.k_proj = nn.ModuleList(
            [
                nn.Linear(decoder_hidden_dim, decoder_hidden_dim)
                for _ in range(self.num_layers)
            ]
        )
        self.v_proj = nn.ModuleList(
            [
                nn.Linear(decoder_hidden_dim, decoder_hidden_dim)
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, latent: torch.FloatTensor) -> DynamicCache:
        """
        Transform latent to be decoder-compatible.

        Args:
            latent: [batch, n_latents, latent_dim]
            - custom latent representation

        Returns:
            adapted: List of (K, V) tuples (one tuple per decoder layer).
                each K/V is of shape :
                [batch, deocder_n_heads, num_prefix_tokens, decoder_head_dim]
        """

        B = latent.size(0)

        # Cross-attend: prefix queries extract P tokens from latent
        prefix = self.cross_attention(
            query=self.prefix_queries.expand(B, -1, -1), key=latent, value=latent
        )  # [B, P, latent_dim]

        # Project to decoder hidden size
        x = self.latent_proj(prefix)
        x = self.norm(x + self.adapter(x))  # residual in decoder space

        # Split into K/V
        kv_cache = []

        for l in range(self.num_layers):
            k = (
                self.k_proj[l](x)
                .view(B, self.num_prefix_tokens, self.decoder_n_heads, self.d_head)
                .transpose(1, 2)
            )
            v = (
                self.v_proj[l](x)
                .view(B, self.num_prefix_tokens, self.decoder_n_heads, self.d_head)
                .transpose(1, 2)
            )
            kv_cache.append((k, v))

        return DynamicCache.from_legacy_cache(kv_cache)


class Decoder(nn.Module):
    """
    Wrapper around pretrained decoder with adapter for custom latents.
    """

    def __init__(
        self,
        pretrained_decoder: nn.Module,
        pretrained_config: PretrainedConfig,
        latent_dim: int,
        n_heads: int = 8,
        num_prefix_tokens: int = 16,
        adapter_dim: Optional[int] = None,
        dropout: float = 0.1,
        max_target_length: int = 1024,
    ):
        super().__init__()

        self.pretrained_decoder = pretrained_decoder
        self.latent_dim = latent_dim
        self.max_target_length = max_target_length

        # Extract decoder configuration
        self.config = pretrained_config
        self.vocab_size = pretrained_config.vocab_size

        # Get decoder hidden dimension
        if hasattr(self.config, "hidden_size"):
            decoder_hidden_dim = self.config.hidden_size
        elif hasattr(self.config, "d_model"):
            decoder_hidden_dim = self.config.d_model
        elif hasattr(self.config, "n_embed"):
            decoder_hidden_dim = self.config.n_embed
        else:
            raise ValueError("Could not determine decoder hidden dimension from config")

        self.decoder_hidden_dim = decoder_hidden_dim

        # Get decoder num heads.
        if hasattr(self.config, "num_attention_heads"):
            decoder_n_heads = self.config.num_attention_heads
        elif hasattr(self.config, "n_head"):
            decoder_n_heads = self.config.n_head
        else:
            raise ValueError("Could not determine decoder num heads from config")

        self.decoder_n_heads = decoder_n_heads

        # Get number of layers.
        if hasattr(self.config, "n_layer"):
            num_layers = self.config.n_layer
        elif hasattr(self.config, "num_hidden_layers"):
            num_layers = self.config.num_hidden_layers
        else:
            raise ValueError("Could not determine decoder num layers from config")

        # Create adapter
        self.adapter = CrossAttentionAdapter(
            latent_dim=latent_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            decoder_n_heads=decoder_n_heads,
            decoder_n_layers=num_layers,
            n_heads=n_heads,
            num_prefix_tokens=num_prefix_tokens,
            adapter_dim=adapter_dim,
            dropout=dropout,
        )

    def forward(
        self,
        latent: torch.Tensor,
        target_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with target tokens (teacher forcing).

        Args:
            latent: [batch, n_latents, latent_dim] - encoded representation
            target_tokens: [batch, seq_len] - target sequence for teacher forcing
            attention_mask: Optional attention mask for target tokens
            **kwargs: Additional arguments passed to decoder

        Returns:
            logits: [batch, seq_len, vocab_size] - output logits
        """
        kv_cache = self.adapter(latent)
        prefix_len = kv_cache.get_seq_length()
        T = target_tokens.size(1)
        cache_position = torch.arange(
            prefix_len, prefix_len + T, device=target_tokens.device
        )

        outputs = self.pretrained_decoder(
            input_ids=target_tokens,
            past_key_values=kv_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
            **kwargs,
        )

        return outputs.logits

    def generate(
        self,
        latent: torch.Tensor,
        max_length: int = 512,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 0.9,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text from latent representation.

        Args:
            latent: [batch, n_latents, latent_dim] - encoded representation
            max_length: Maximum generation length
            do_sample: Whether to sample or use greedy decoding
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            bos_token_id: Beginning-of-sequence token ID
            **kwargs: Additional generation arguments

        Returns:
            generated_ids: [batch, gen_len] - generated token IDs
        """
        # Set default token IDs if not provided
        if pad_token_id is None:
            pad_token_id = getattr(self.config, "pad_token_id", 0)
        if eos_token_id is None:
            eos_token_id = getattr(self.config, "eos_token_id", 1)

        bos_token_id = getattr(self.config, "bos_token_id", None)

        if bos_token_id is None:
            raise ValueError("Please provide bos_token_id for generation")

        kv_cache = self.adapter(latent)
        prefix_len = kv_cache.get_seq_length()
        bsz = latent.size(0)
        input_ids = torch.full(
            (bsz, 1), bos_token_id, device=latent.device, dtype=torch.long
        )
        cache_position = torch.arange(prefix_len, prefix_len + 1, device=latent.device)

        generated = self.pretrained_decoder.generate(
            input_ids=input_ids,
            past_key_values=kv_cache,
            cache_position=cache_position,
            use_cache=True,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        return generated

    def get_num_parameters(self) -> dict:
        """Get parameter counts for analysis."""
        total_params = sum(p.numel() for p in self.parameters())
        adapter_params = sum(p.numel() for p in self.adapter.parameters())
        decoder_params = sum(p.numel() for p in self.pretrained_decoder.parameters())

        return {
            "total_parameters": total_params,
            "adapter_parameters": adapter_params,
            "pretrained_decoder_parameters": decoder_params,
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }
