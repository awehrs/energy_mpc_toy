from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from models.attention import Attention


class Downsampler(nn.Module):
    """Perceiver encoder."""

    def __init__(
        self,
        d_model: int,
        d_input: Optional[int] = None,
        n_latents: int = 64,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        self.query = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)

        self.cross_attention = Attention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            query_dim=d_model,
            key_dim=d_input if d_input is not None else d_model,
            value_dim=d_input,
        )

        self.pooling_norm = nn.RMSNorm(d_model)

        # init
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        batch_size, _, _ = inputs.shape

        query = self.query.expand(batch_size, -1, -1)

        latent = self.cross_attention(
            query=query,
            key=inputs,
            value=inputs,
            attn_mask=attention_mask,
        )

        latent = self.pooling_norm(latent)

        return latent


class LanguageSensor(nn.Module):
    """
    Pretrained LLM + deterministic downsampler.
    """

    def __init__(
        self,
        model_name: str,
        d_model: int = 1024,
        n_precept_latents: int = 64,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.pretrained_llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # Get decoder hidden dimension
        if hasattr(self.config, "hidden_size"):
            llm_hidden_dim = self.pretrained_llm.config.hidden_size
        elif hasattr(self.config, "d_model"):
            llm_hidden_dim = self.pretrained_llm.config.d_model
        elif hasattr(self.config, "n_embed"):
            llm_hidden_dim = self.pretrained_llm.config.n_embed
        else:
            raise ValueError("Could not determine decoder hidden dimension from config")

        self.downsampler = Downsampler(
            d_model=d_model,
            d_input=llm_hidden_dim,
            n_latents=n_precept_latents,
            n_heads=n_heads,
            dropout=dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        logits = self.pretrained_llm(input_ids, attention_mask).logits

        return self.downsampler(logits)
