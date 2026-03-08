import abc
from typing import Optional

import einops
import torch
import torch.nn as nn

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
except ImportError:
    flex_attention, create_block_mask = None, None

from models.attention import MLP


class Predictor(abc.ABC, nn.Module):
    pass


class JEPAPredictor(Predictor):
    """Predicts next-step precept latents using banded flex attention.

    Concatenates state (from memory) and current action latents per step,
    then applies masked attention where step t attends to steps {t-1, t}.

    Note: flex_attention requires torch.compile for fused-kernel performance.
    Compile the model or the forward method externally.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int = 4,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": nn.RMSNorm(d_model, eps=norm_eps),
                        "q_proj": nn.Linear(d_model, d_model, bias=False),
                        "k_proj": nn.Linear(d_model, d_model, bias=False),
                        "v_proj": nn.Linear(d_model, d_model, bias=False),
                        "out_proj": nn.Linear(d_model, d_model, bias=False),
                        "ffn_norm": nn.RMSNorm(d_model, eps=norm_eps),
                        "ffn": MLP(d_model),
                    }
                )
                for _ in range(n_layers)
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    @staticmethod
    def build_block_mask(traj_mask: torch.Tensor, N: int, device: torch.device):
        """Build flex_attention block mask (call outside torch.compile).

        Args:
            traj_mask: [B, T] — True for valid steps
            N: latents per step (Ns + Na)
            device: target device
        """
        B, T = traj_mask.shape
        S = T * N

        def mask_mod(b, h, q_idx, kv_idx):
            q_step = q_idx // N
            kv_step = kv_idx // N
            same_or_prev = (kv_step == q_step) | (kv_step == q_step - 1)
            q_valid = traj_mask[b, q_step]
            kv_valid = traj_mask[b, kv_step]
            return same_or_prev & q_valid & kv_valid

        return create_block_mask(
            mask_mod, B=B, H=None, Q_LEN=S, KV_LEN=S, device=device,
        )

    @classmethod
    @torch.no_grad()
    def warmup_flex_attention(cls, device, d_model, n_heads, n_state_latents, n_action_latents, min_T=2, max_T=14):
        """Pre-compile flex_attention and create_block_mask kernels for expected T values."""
        N = n_state_latents + n_action_latents
        B = 2
        H = n_heads
        D = d_model // n_heads
        dtype = torch.bfloat16
        for T in range(min_T, max_T + 1):
            S = T * N
            traj_mask = torch.ones(B, T, dtype=torch.bool, device=device)
            block_mask = cls.build_block_mask(traj_mask, N, device)
            q = torch.zeros(B, H, S, D, device=device, dtype=dtype)
            k = torch.zeros(B, H, S, D, device=device, dtype=dtype)
            v = torch.zeros(B, H, S, D, device=device, dtype=dtype)
            flex_attention(q, k, v, block_mask=block_mask)
            torch.cuda.synchronize()

    def predict_step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Single-step prediction for differentiable rollouts.

        Wraps forward() with T=1. flex_attention compiles once per (B, N) shape.

        Args:
            state:  [B, Ns, d]
            action: [B, Na, d]
        Returns:
            [B, Ns, d]
        """
        B, Ns, _ = state.shape
        N = Ns + action.shape[1]

        state_1  = state.unsqueeze(1)   # [B, 1, Ns, d]
        action_1 = action.unsqueeze(1)  # [B, 1, Na, d]

        traj_mask  = torch.ones(B, 1, dtype=torch.bool, device=state.device)
        block_mask = self.build_block_mask(traj_mask, N, state.device)

        out = self.forward(state=state_1, block_mask=block_mask, action=action_1)
        return out[:, 0, :Ns, :]  # [B, Ns, d]

    def forward(
        self,
        state: torch.Tensor,
        block_mask,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            state:      [B, T, Ns, d]  — memory output (or raw precept latents for unconditional)
            block_mask: BlockMask from build_block_mask (computed outside compile)
            action:     [B, T, Na, d]  — current action latents (optional; None for unconditional)
        Returns:
            [B, T, Ns+Na, d] if action provided, [B, T, Ns, d] otherwise
        """
        if action is not None:
            x_4d = torch.cat([state, action], dim=2)  # [B, T, Ns+Na, d]
        else:
            x_4d = state  # [B, T, Ns, d]
        B, T, N, d = x_4d.shape
        H = self.n_heads
        D = self.head_dim
        S = T * N

        # Flatten steps × latents: [B, T*(Ns+Na), d]
        x = einops.rearrange(x_4d, "b t n d -> b (t n) d")

        for layer in self.layers:
            residual = x
            h = layer["norm"](x)

            q = layer["q_proj"](h).view(B, S, H, D).transpose(1, 2)
            k = layer["k_proj"](h).view(B, S, H, D).transpose(1, 2)
            v = layer["v_proj"](h).view(B, S, H, D).transpose(1, 2)

            attn_out = flex_attention(q, k, v, block_mask=block_mask)
            attn_out = attn_out.transpose(1, 2).reshape(B, S, d)

            x = layer["out_proj"](attn_out) + residual

            residual = x
            x = layer["ffn"](layer["ffn_norm"](x)) + residual

        return einops.rearrange(x, "b (t n) d -> b t n d", t=T, n=N)
