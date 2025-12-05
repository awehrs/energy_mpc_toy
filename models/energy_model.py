import torch
import torch.nn as nn

from models.attention import Attention


class EnergyModel(nn.Module):
    """
    Transformer-based Energy Model that evaluates latent representation quality.
    Assigns low energy to latents that contain sufficient task-relevant information.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 1024,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.energy_query = nn.Parameter(torch.randn(1, d_model) * 0.02)

        self.cross_attention = Attention(
            d_model=d_model,
            n_heads=n_heads,
            query_dim=d_model,
            key_dim=input_dim,
            value_dim=input_dim,
        )

        self.pooling_norm = nn.LayerNorm(d_model)

        # Energy prediction head - outputs a single scalar
        self.energy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
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
            energy: [batch] - scalar energy for each example
        """
        batch_size, _, _ = latent.shape

        # Single learned query attends to all processed latent tokens
        energy_query = self.energy_query.expand(
            batch_size, -1, -1
        )  # [batch, 1, d_model]

        # Cross-attention: query attends to processed latents
        pooled = self.cross_attention(
            query=energy_query,
            key=latent,
            value=latent,
        )

        # Layer norm and squeeze
        pooled = self.pooling_norm(pooled)
        pooled = pooled.squeeze(1)  # [batch, d_model]

        # Predict energy (single scalar per batch element)
        energy = self.energy_head(pooled).squeeze(-1)  # [batch]

        return energy
