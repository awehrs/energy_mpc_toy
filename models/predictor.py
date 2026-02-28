import abc
import time
import logging
from typing import Optional

import einops
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

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

    @torch.no_grad()
    def warmup_flex_attention(self, device, n_state_latents, n_action_latents, min_T=2, max_T=14, B=2):
        """Pre-compile flex_attention kernels for all expected T values."""
        N = n_state_latents + n_action_latents
        logger.info(f"Warming up flex_attention for T in [{min_T}, {max_T}] (N={N})...")
        for T in range(min_T, max_T + 1):
            t0 = time.time()
            dtype = next(self.parameters()).dtype
            state = torch.zeros(B, T, n_state_latents, self.d_model, device=device, dtype=dtype)
            action = torch.zeros(B, T, n_action_latents, self.d_model, device=device, dtype=dtype)
            traj_mask = torch.ones(B, T, dtype=torch.bool, device=device)
            self.forward(state, traj_mask, action=action)
            torch.cuda.synchronize()
            logger.info(f"  T={T} (S={T * N}): {time.time() - t0:.2f}s")
        logger.info("Flex attention warmup done.")

    def _make_mask_mod(self, traj_mask, N):
        """Return a mask_mod closure for create_block_mask.

        Pattern: position at step t attends to steps {t-1, t} only,
        and both query/key positions must be valid (non-padded) steps.
        """

        def mask_mod(b, h, q_idx, kv_idx):
            q_step = q_idx // N
            kv_step = kv_idx // N
            same_or_prev = (kv_step == q_step) | (kv_step == q_step - 1)
            q_valid = traj_mask[b, q_step]
            kv_valid = traj_mask[b, kv_step]
            return same_or_prev & q_valid & kv_valid

        return mask_mod

    def forward(
        self,
        state: torch.Tensor,
        traj_mask: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            state:     [B, T, Ns, d]  — memory output (or raw precept latents for unconditional)
            traj_mask: [B, T]
            action:    [B, T, Na, d]  — current action latents (optional; None for unconditional)
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

        # Build flex attention block mask
        t0 = time.time()
        mask_mod = self._make_mask_mod(traj_mask, N)
        block_mask = create_block_mask(
            mask_mod,
            B=B,
            H=None,
            Q_LEN=S,
            KV_LEN=S,
            device=state.device,
        )
        torch.cuda.synchronize()
        logger.info(f"    create_block_mask: {time.time() - t0:.3f}s (B={B}, T={T}, S={S})")

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
