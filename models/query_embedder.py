import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import Attention


class QueryEmbedder(nn.Module):
    """
    Cross-attention based model that converts latent actions to retrieval queries.
    """

    def __init__(
        self,
        input_dim: int,
        index_dim: int,
        d_model: int = 1024,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.retrieval_pooler = nn.Parameter(torch.randn(1, d_model) * 0.02)

        self.cross_attention = Attention(
            d_model=d_model,
            n_heads=n_heads,
            query_dim=d_model,
            key_dim=input_dim,
            value_dim=input_dim,
        )

        self.pooling_norm = nn.LayerNorm(d_model)

        # Energy prediction head - outputs a single scalar
        self.retrieval_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, index_dim),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: [batch, n_latents, input_dim] - latent representation to evaluate

        Returns:
            retrieval_query: [batch, index_dim] - query for dense vector database
        """
        batch_size, _, _ = latent.shape

        # Single learned query attends to all processed action tokens.
        retrieval_pooler = self.retrieval_pooler.expand(
            batch_size, -1, -1
        )  # [batch, 1, d_model]

        # Cross-attention: query attends to processed action tokens.
        pooled = self.cross_attention(
            query=retrieval_pooler,
            key=latent,
            value=latent,
        )

        # Layer norm and squeeze
        pooled = self.pooling_norm(pooled)
        pooled = pooled.squeeze(1)  # [batch, d_model]

        # Predict energy (single scalar per batch element)
        retrieval_query = self.retrieval_head(pooled)  # [batch, index_dim]

        # Normalize for cosine similarity search
        retrieval_query = F.normalize(retrieval_query, p=2, dim=-1)

        return retrieval_query
