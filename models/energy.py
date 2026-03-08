import torch
import torch.nn as nn

from models.attention import Attention


class CompletionModel(nn.Module):
    """Binary classifier: P(terminal | state) ∈ (0, 1).

    Pools state latents via cross-attention → scalar logit.
    Trained with BCE: s_terminal → 1, all other states → 0.
    Used by get_energy_targets as the stop cost: -log P(terminal | state).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.norm_kv = nn.RMSNorm(d_model, eps=norm_eps)
        self.cross_attn = Attention(
            d_model=d_model,
            n_heads=n_heads,
            is_causal=False,
            is_cross_attention=True,
        )
        self.norm_out = nn.RMSNorm(d_model, eps=norm_eps)
        self.out_proj = nn.Linear(d_model, 1, bias=False)
        nn.init.zeros_(self.out_proj.weight)  # start neutral

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, Ns, d]
        Returns:
            logits: [B]  (apply sigmoid for probabilities)
        """
        B = state.shape[0]
        q = self.query.expand(B, -1, -1)
        pooled = self.cross_attn(query=q, key=self.norm_kv(state)).squeeze(1)
        return self.out_proj(self.norm_out(pooled)).squeeze(-1)

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """P(terminal | state) ∈ (0, 1). Shape: [B]"""
        return torch.sigmoid(self.forward(state))

    def loss(
        self,
        terminal_states: torch.Tensor,
        nonterminal_states: torch.Tensor,
    ) -> torch.Tensor:
        """BCE loss on balanced terminal / non-terminal batch.

        Args:
            terminal_states:    [B, Ns, d] — last valid step of each trajectory
            nonterminal_states: [M, Ns, d] — randomly sampled non-terminal steps
        Returns:
            scalar BCE loss
        """
        states = torch.cat([terminal_states, nonterminal_states], dim=0)
        labels = torch.cat(
            [
                torch.ones(len(terminal_states), device=states.device),
                torch.zeros(len(nonterminal_states), device=states.device),
            ]
        )
        return nn.functional.binary_cross_entropy_with_logits(
            self.forward(states), labels
        )


class Energy(nn.Module):
    """Energy model: pools state latents via cross-attention → scalar energy.

    Low energy = state is close to task completion.
    Terminal state (s_terminal) is trained to have energy ≈ 0.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        # Single learned query pools over Ns state latent tokens
        self.energy_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.norm_kv = nn.RMSNorm(d_model, eps=norm_eps)

        self.cross_attn = Attention(
            d_model=d_model,
            n_heads=n_heads,
            is_causal=False,
            is_cross_attention=True,
        )

        self.norm_out = nn.RMSNorm(d_model, eps=norm_eps)
        self.out_proj = nn.Linear(d_model, 1, bias=False)

        nn.init.normal_(self.out_proj.weight, std=0.02)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, Ns, d]
        Returns:
            energy: [B]
        """
        B = state.shape[0]

        q = self.energy_query.expand(B, -1, -1)  # [B, 1, d]
        pooled = self.cross_attn(query=q, key=self.norm_kv(state))  # [B, 1, d]
        pooled = pooled.squeeze(1)  # [B, d]

        return self.out_proj(self.norm_out(pooled)).squeeze(-1)  # [B]
