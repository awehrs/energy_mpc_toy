import abc

import einops
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM


class Sensor(abc.ABC, nn.Module):
    pass


class LanguageSensor(Sensor):
    """Pretrained LLM encoder with linear projection.

    Returns packed hidden states — downsampling is handled externally.
    """

    def __init__(
        self,
        model_name: str,
        d_model: int = 1024,
    ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)

        # Get encoder hidden dimension
        if hasattr(self.config, "hidden_size"):
            hidden_dim = self.config.hidden_size
        elif hasattr(self.config, "d_model"):
            hidden_dim = self.config.d_model
        elif hasattr(self.config, "n_embed"):
            hidden_dim = self.config.n_embed
        else:
            raise ValueError("Could not determine encoder hidden dimension from config")

        self.encoder = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            attn_implementation="flash_attention_2",
        )

        self.encoder.requires_grad_(False)

        self.projection = (
            nn.Linear(hidden_dim, d_model) if hidden_dim != d_model else nn.Identity()
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:    [total_tokens]  — packed flat token ids
            position_ids: [total_tokens]  — per-token position ids
        Returns:
            [total_tokens, d]  — projected encoder hidden states
        """
        input_ids = einops.rearrange(input_ids, "t -> 1 t")
        position_ids = einops.rearrange(position_ids, "t -> 1 t")

        hidden_states = self.encoder.model(
            input_ids=input_ids,
            position_ids=position_ids,
        ).last_hidden_state

        hidden_states = self.projection(hidden_states)

        return einops.rearrange(hidden_states, "1 t d -> t d")
