import math
from typing import Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import Attention, MLP


class Policy(nn.Module):
    pass


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by a two-layer MLP.

    Args:
        sin_dim:      dimension of sinusoidal encoding (default 256)
        time_emb_dim: output dimension passed to AdaLN layers
    """

    def __init__(self, sin_dim: int = 256, time_emb_dim: int = 1024):
        super().__init__()
        assert sin_dim % 2 == 0
        self.sin_dim = sin_dim
        self.mlp = nn.Sequential(
            nn.Linear(sin_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] — timestep values in [0, 1]
        Returns:
            [B, time_emb_dim]
        """
        half = self.sin_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * freqs[None]                        # [B, half]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)      # [B, sin_dim]
        return self.mlp(emb)                                    # [B, time_emb_dim]


class AdaLN(nn.Module):
    """Adaptive LayerNorm: scale and shift conditioned on a time embedding.

    Initialised so the transform is identity at the start of training
    (zero-init on the projection).
    """

    def __init__(self, d_model: int, time_emb_dim: int, norm_eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=norm_eps)
        self.proj = nn.Linear(time_emb_dim, 2 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:        [B, N, d]
            time_emb: [B, time_emb_dim]
        Returns:
            [B, N, d]
        """
        scale, shift = self.proj(time_emb).chunk(2, dim=-1)    # [B, d] each
        scale = scale.unsqueeze(1)                              # [B, 1, d]
        shift = shift.unsqueeze(1)
        return self.norm(x) * (1 + scale) + shift


class PolicyBlock(nn.Module):
    """Single flow-matching transformer block.

    Per-layer structure:
        AdaLN → self-attn  (among Na action tokens)      → residual
        AdaLN → cross-attn (action tokens attend to state) → residual
        AdaLN → FFN                                        → residual
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        time_emb_dim: int,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.norm_self  = AdaLN(d_model, time_emb_dim, norm_eps)
        self.self_attn  = Attention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            is_causal=False,
            is_cross_attention=False,
        )

        self.norm_cross = AdaLN(d_model, time_emb_dim, norm_eps)
        self.cross_attn = Attention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            is_causal=False,
            is_cross_attention=True,
        )

        self.norm_ffn = AdaLN(d_model, time_emb_dim, norm_eps)
        self.ffn      = MLP(d_model)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:        [B, Na, d]  — noisy action latents
            state:    [B, Ns, d]  — current step state (from memory)
            time_emb: [B, time_emb_dim]
        Returns:
            [B, Na, d]
        """
        x = x + self.self_attn(query=self.norm_self(x, time_emb))
        x = x + self.cross_attn(query=self.norm_cross(x, time_emb), key=state)
        x = x + self.ffn(self.norm_ffn(x, time_emb))
        return x


class FlowPolicy(Policy):
    """Conditional flow matching policy.

    Learns a velocity field v(z_t, t, state) that transports noise → action latents.

    Training (OT-CFM):
        x_0 ~ N(0, I),  x_1 = target action latent
        t   ~ Uniform(0, 1)
        x_t = (1 - t) * x_0 + t * x_1      (linear / OT path)
        u_t = x_1 - x_0                      (constant velocity target)
        loss = MSE(v(x_t, t, state), u_t)

    Inference:
        z_0 ~ N(0, I)
        Euler: z_{i+1} = z_i + (1/n_steps) * v(z_i, i/n_steps, state)
        z_1 ≈ action latent → actuator decodes to tokens
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        n_action_latents: int,
        time_emb_dim: int = 1024,
        sin_dim: int = 256,
        n_integration_steps: int = 10,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.n_action_latents   = n_action_latents
        self.n_integration_steps = n_integration_steps
        self.d_model             = d_model

        self.time_emb = TimestepEmbedding(sin_dim=sin_dim, time_emb_dim=time_emb_dim)

        self.state_norm = nn.LayerNorm(d_model, eps=norm_eps)

        self.blocks = nn.ModuleList([
            PolicyBlock(
                d_model=d_model,
                n_heads=n_heads,
                time_emb_dim=time_emb_dim,
                dropout=dropout,
                norm_eps=norm_eps,
            )
            for _ in range(n_layers)
        ])

        self.out_norm = AdaLN(d_model, time_emb_dim, norm_eps)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity field.

        Args:
            z_t:   [B, Na, d]  — noisy action latents at time t
            t:     [B]         — timestep in [0, 1]
            state: [B, Ns, d]  — current step state from memory model
        Returns:
            velocity: [B, Na, d]
        """
        time_emb = self.time_emb(t)
        state    = self.state_norm(state)

        x = z_t
        for block in self.blocks:
            x = block(x, state, time_emb)

        return self.out_proj(self.out_norm(x, time_emb))

    def flow_matching_loss(
        self,
        action_latents: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """OT-CFM training loss.

        Args:
            action_latents: [B, Na, d]  — target action latents (x_1)
            state:          [B, Ns, d]  — conditioning state
        Returns:
            scalar MSE loss
        """
        x_1 = action_latents
        x_0 = torch.randn_like(x_1)
        t   = torch.rand(x_1.shape[0], device=x_1.device, dtype=x_1.dtype)

        t_e = t[:, None, None]                          # [B, 1, 1] for broadcast
        x_t = (1 - t_e) * x_0 + t_e * x_1             # interpolated sample
        u_t = x_1 - x_0                                # target velocity

        v = self.forward(x_t, t, state)
        return F.mse_loss(v, u_t)

    def sample(
        self,
        state: torch.Tensor,
        num_samples: int = 1,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample action latents via Euler integration.

        Args:
            state:       [B, Ns, d]
            num_samples: k — number of independent samples per state
            n_steps:     integration steps (defaults to self.n_integration_steps)
        Returns:
            [B, k, Na, d]
        """
        n_steps = n_steps or self.n_integration_steps
        B, _, d = state.shape

        # Repeat state for k samples
        state_rep = einops.repeat(state, "b n d -> (b k) n d", k=num_samples)

        # Sample noise
        z = torch.randn(
            B * num_samples, self.n_action_latents, d,
            device=state.device, dtype=state.dtype,
        )

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full(
                (B * num_samples,), i * dt,
                device=state.device, dtype=state.dtype,
            )
            z = z + dt * self.forward(z, t, state_rep)

        return einops.rearrange(z, "(b k) n d -> b k n d", b=B, k=num_samples)
