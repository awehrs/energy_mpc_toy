import abc
from typing import Optional

import einops
import torch
import torch.nn as nn

try:
    from fla.layers import GatedDeltaNet
except ImportError:
    GatedDeltaNet = None

from models.attention import Attention, TransformerBlock, MLP


class Memory(abc.ABC, nn.Module):

    def update(self, state: torch.Tensor, action: torch.Tensor):
        pass

    def read(self, state: torch.Tensor, action: torch.Tensor):
        pass


class GatedDeltaMemoryBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_mixer_layers: int = 2,
        norm_eps: float = 1e-5,
        layer_idx: int = None,
        **gdn_kwargs,
    ):
        super().__init__()

        self.intra_step_mixer = TransformerBlock(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_mixer_layers,
            is_causal=False,
            is_cross_attention=False,
        )

        self.gdn_norm = nn.RMSNorm(d_model, eps=norm_eps)

        self.gdn = GatedDeltaNet(
            hidden_size=d_model,
            layer_idx=layer_idx,
            **gdn_kwargs,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, T, N, d]  (N = n_state_latents)
        Returns:
            [B, T, N, d]
        """
        B, T, N, d = state.shape

        # Intra-step mixing: non-causal self-attention among latents per step.
        # TransformerBlock has internal residual connections.
        mixed = einops.rearrange(state, "b t n d -> (b t) n d")
        mixed = self.intra_step_mixer(q=mixed)
        state = einops.rearrange(mixed, "(b t) n d -> b t n d", b=B, t=T)

        # Lane-parallel GDN: each latent lane processed causally across time.
        residual = state
        state = self.gdn_norm(state)
        laned = einops.rearrange(state, "b t n d -> (b n) t d")
        laned, _, _ = self.gdn(laned)
        state = einops.rearrange(laned, "(b n) t d -> b t n d", b=B, n=N)

        return state + residual


class GatedDeltaMemory(Memory):

    def __init__(
        self,
        d_model: int,
        n_state_latents: int,
        n_heads: int,
        n_blocks: int = 4,
        n_mixer_layers: int = 2,
        norm_eps: float = 1e-5,
        **gdn_kwargs,
    ):
        super().__init__()

        self.n_state_latents = n_state_latents

        # Learned fusion queries
        self.fusion_latents = nn.Parameter(
            torch.randn(n_state_latents, d_model) * 0.02
        )

        # Fusion: cross-attend from learned latents to concat(precepts, actions)
        self.fusion_norm_kv = nn.RMSNorm(d_model, eps=norm_eps)
        self.fusion_attn = Attention(
            d_model=d_model,
            n_heads=n_heads,
            is_causal=False,
            is_cross_attention=True,
        )
        self.fusion_ffn_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.fusion_ffn = MLP(d_model)

        # Memory blocks
        self.blocks = nn.ModuleList(
            [
                GatedDeltaMemoryBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_mixer_layers=n_mixer_layers,
                    norm_eps=norm_eps,
                    layer_idx=i,
                    **gdn_kwargs,
                )
                for i in range(n_blocks)
            ]
        )

    def forward(
        self,
        precepts: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            precepts:        [B, T, n_latents, d]
            previous_action: [B, T, n_latents, d]
        Returns:
            [B, T, n_state_latents, d]
        """
        B, T, n_latents, d = precepts.shape

        # --- Fuse precepts + actions via Perceiver-style cross-attention ---

        # KV: concat along latent dim
        kv = torch.cat([precepts, previous_action], dim=2)  # [B, T, 2*n_latents, d]
        kv = einops.rearrange(kv, "b t n d -> (b t) n d")
        kv = self.fusion_norm_kv(kv)

        # Q: learned fusion latents, repeated per (batch, step) pair
        q = einops.repeat(self.fusion_latents, "n d -> (b t) n d", b=B, t=T)

        # Cross-attention (batched 3D)
        state = self.fusion_attn(query=q, key=kv)  # [B*T, n_state_latents, d]
        state = state + q  # residual

        # FFN
        state = self.fusion_ffn(self.fusion_ffn_norm(state)) + state

        state = einops.rearrange(state, "(b t) n d -> b t n d", b=B, t=T)

        # --- Process through memory blocks ---

        for block in self.blocks:
            state = block(state)

        return state

    def generate(self, state: torch.Tensor, action: torch.Tensor):
        pass
