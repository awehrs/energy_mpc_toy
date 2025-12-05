from typing import Dict

import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm import Mamba2
from jaxtyping import Bool, Float, Integer

from models.attention import Attention, MLP
from models.sensors import LanguageSensor


class ActionFusion(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_cross_attn_heads: int,
        n_self_attn_heads: int,
        dropout: int,
        query_dim: int,
        kv_dim: int,
    ):
        super().__init__()

        assert n_layers % 2 == 0

        attn_layers = []

        for i in range(n_layers):
            is_cross_attn = (i + 1) % 2 == 0
            key_dim = value_dim = kv_dim if is_cross_attn else query_dim
            n_heads = n_cross_attn_heads if is_cross_attn else n_self_attn_heads

            attn_layers.append(
                nn.ModuleDict(
                    {
                        "norm1": nn.RMSNorm(query_dim),
                        "self_attn": Attention(
                            d_model=query_dim,
                            key_dim=key_dim,
                            value_dim=value_dim,
                            n_heads=n_heads,
                            dropout=dropout,
                        ),
                        "norm2": nn.RMSNorm(query_dim),
                        "ffn": MLP(query_dim),
                    }
                )
            )

        self.layers = nn.ModuleList(attn_layers)

    def foward(
        self,
        actions: Float[torch.Tensor, "batch seq_len dim"],
        precepts: Float[torch.Tensor, "batch seq_len dim"],
    ) -> Float[torch.Tensor, "batch seq_len_dim"]:

        for i, layer in enumerate(self.layers):
            is_cross_attn = (i + 1) % 2 == 0
            precepts = layer(query=precepts, key=actions if is_cross_attn else None)

        return precepts


class DynamicsModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        d_action_latent: int,
        d_precept_latent: int,
        d_ssm_model: int,
        n_precept_latents: int,
        n_ssm_layers: int,
        n_attn_layers: int,
        n_self_attn_heads: int,
        n_cross_attn_heads: int,
        dropout: int,
    ):
        super().__init__()

        self.sensor = LanguageSensor(
            model_name=model_name,
            d_model=d_precept_latent,
            n_precept_latents=n_precept_latents,
            n_heads=n_cross_attn_heads,
            dropout=dropout,
        )

        self.action_fusion = ActionFusion(
            n_layers=n_attn_layers,
            n_cross_attn_heads=n_cross_attn_heads,
            n_self_attn_heads=n_self_attn_heads,
            dropout=dropout,
            query_dim=d_precept_latent,
            kv_dim=d_action_latent,
        )

        self.layers = nn.ModuleList([])

        for _ in range(n_ssm_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "mamba": Mamba2(
                            d_model=d_ssm_model,
                            d_state=64,
                            d_conv=4,
                            expand=2,
                        ),
                        "norm": nn.RMSNorm(d_ssm_model),
                        "dropout": nn.Dropout,
                    }
                )
            )

        self.projection = nn.Linear(d_ssm_model, d_precept_latent)

    def forward(
        self,
        input_ids: Integer[torch.Tensor, "batch num_steps seq_len"],
        attention_mask: Bool[torch.Tensor, "batch num_steps seq_len"],
        action_latents: Float[torch.Tensor, "batch num_steps seq_len n_latents dim"],
    ) -> Dict[str, torch.Tensor]:

        bsz = action_latents.shape[0]

        precept_latents = self.sensor(input_ids, attention_mask)

        precept_latents = rearrange("b n t d -> (b n) t d")

        action_latents = rearrange("b n t d -> (b n) t d")

        z = self.action_fusion(
            actions=action_latents,
            precepts=precept_latents,
        )

        z = rearrange("(b n) t d -> (b t) n d", b=bsz)

        z, h = self.dynamics_model(z)

        y = rearrange(z, "(b t) n d -> b n t d", b=bsz)

        return {
            "output": y,
            "state": h,
        }
