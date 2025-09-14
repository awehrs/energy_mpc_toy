import torch
import torch.nn as nn

from models.utils import Attention


class ActionProjection(nn.Module):
    """
    Projects retrieval queries [batch, n_action_tokens, index_dim] into forward model latent space
        [batch, n_latent_action_tokens, d_model].

    Used when actions are hard-coded queries.
    """

    def __init__(
        self,
        index_dim: int,
        d_model: int,
        n_heads: int = 8,
        n_action_tokens: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        self.action_query = nn.Parameter(torch.randn(n_action_tokens, d_model) * 0.02)

        self.cross_attention = Attention(
            d_model=d_model,
            n_heads=n_heads,
            query_dim=d_model,
            key_dim=index_dim,
            value_dim=index_dim,
        )

        self.pooling_norm = nn.LayerNorm(d_model)

        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, d_model),
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

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: [batch, n_actions_tokens, index_dim]

        Returns:
            projected: [batch, n_latent_action_tokens, d_model]
        """
        batch_size, _, _ = queries.shape

        # Single learned query attends to all processed action tokens.
        action_query = self.action_query.expand(
            batch_size, -1, -1
        )  # [batch, n_latent_action_tokens, d_model]

        # Cross-attention: query attends to processed action tokens.
        pooled = self.cross_attention(
            query=action_query,
            key=queries,
            value=queries,
        )

        # Layer norm and squeeze
        pooled = self.pooling_norm(pooled)

        # Predict energy (single scalar per batch element)
        latent_action_tokens = self.action_head(
            pooled
        )  # [batch, n_latent_action_tokens, d_model]

        return latent_action_tokens
