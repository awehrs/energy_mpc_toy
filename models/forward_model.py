from typing import Tuple

import einops
import torch
import torch.nn as nn

from models.utils import Attention

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False


class ForwardModel(nn.Module):
    """
    Sequence model over compressed inputs and latent action intentions.
    Each step = [D pseudo doc tokens][A action tokens].
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 16,
        n_layers: int = 4,
        dropout: float = 0.1,
        n_bottleneck_tokens: int = 16,
        n_action_tokens_per_step: int = 1,
        max_steps: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.D = n_bottleneck_tokens
        self.A = n_action_tokens_per_step
        self.S = self.D + self.A
        self.max_steps = max_steps

        self.flex_attention_enabled = False
        self.block_mask_full = None

        # Build mask once for maximum sequence length
        self._setup_flex_attention()

        # Transformer stack
        self.temporal_transformer = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": Attention(
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

        # init
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _setup_flex_attention(self):
        if FLEX_ATTENTION_AVAILABLE:

            def mask_mod(b, h, q_idx, kv_idx):
                # Integer division and modulus in Torch
                q_step = torch.div(q_idx, self.S, rounding_mode="floor")
                q_off = q_idx % self.S
                kv_step = torch.div(kv_idx, self.S, rounding_mode="floor")
                kv_off = kv_idx % self.S

                is_q_pseudo = q_off < self.D
                is_q_action = q_off >= self.D
                is_kv_pseudo = kv_off < self.D
                same_step = q_step == kv_step

                # Start with causal constraint: can only attend to same or previous steps
                allow = kv_step <= q_step

                # Within the same step, apply specific rules:
                when_same_step = same_step & allow

                # Doc tokens CANNOT attend to action tokens in same step
                doc_to_action_same_step = is_q_pseudo & (~is_kv_pseudo) & when_same_step

                # Combine: allow causal + same-step rules, but block doc->action same-step
                allow = allow & (~doc_to_action_same_step)

                return allow

            seq_len = self.max_steps * self.S

            device = "cuda" if torch.cuda.is_available() else "cpu"

            self.block_mask_full = create_block_mask(
                mask_mod=mask_mod,
                B=None,
                H=None,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=device,
                BLOCK_SIZE=1,
            )

            self.flex_attention_enabled = True
        else:
            self.block_mask = None
            self.flex_attention_enabled = False
            print("FlexAttention not available, falling back to standard attention")

    def _apply(self, fn):
        # override nn.Module._apply so that .to() or .cuda() triggers mask rebuild
        super()._apply(fn)
        if FLEX_ATTENTION_AVAILABLE:
            self._setup_flex_attention()
        return self

    def _flex_attention_forward(self, attention_layer: nn.Module, x: torch.Tensor):
        batch_size, seq_len, d_model = x.shape

        Q = attention_layer.q_proj(x)
        K = attention_layer.k_proj(x)
        V = attention_layer.v_proj(x)

        Q = Q.view(
            batch_size, seq_len, attention_layer.n_heads, attention_layer.head_dim
        ).transpose(1, 2)
        K = K.view(
            batch_size, seq_len, attention_layer.n_heads, attention_layer.head_dim
        ).transpose(1, 2)
        V = V.view(
            batch_size, seq_len, attention_layer.n_heads, attention_layer.head_dim
        ).transpose(1, 2)

        attn_output = flex_attention(Q, K, V, block_mask=self.block_mask_full)

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )
        return attention_layer.out_proj(attn_output)

    def forward(
        self,
        doc_latents: torch.Tensor,  # [B, steps+1, D, d_model] (includes question step)
        action_latents: torch.Tensor,  # [B, steps, A, d_model] (no action for question)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            doc_latents: [batch, steps+1, n_docs_per_step * n_doc_latents_per_step, d_model] (includes question)
            action_latents: [batch, steps, n_action_tokens_per_step, d_model] (retrieval steps only)
        Returns:
            docs_out: [batch, steps+1, n_latent_doc_tokens_per_step, d_model] (includes question)
            acts_out: [batch, steps+1, n_action_tokens_per_step, d_model] (padded with zeros for question)

        """
        batch_size, steps_plus_one, doc_tokens, d_model = doc_latents.shape

        # Both inputs should already have the same number of steps (steps+1)
        latent = torch.cat(
            [doc_latents, action_latents], dim=2
        )  # [B, steps+1, D+A, d_model]
        latent = einops.rearrange(latent, "b n t h -> b (n t) h")

        for layer in self.temporal_transformer:
            if self.flex_attention_enabled:
                attn_out = self._flex_attention_forward(layer["attn"], latent)
            else:
                attn_out = layer["attn"](latent)
            latent = layer["norm1"](latent + attn_out)

            ffn_out = layer["ffn"](latent)
            latent = layer["norm2"](latent + ffn_out)

        latent = einops.rearrange(latent, "b (n t) h -> b n t h", t=self.S)
        docs_out = latent[:, :, : self.D, :]
        acts_out = latent[:, :, self.D :, :]

        return docs_out, acts_out
