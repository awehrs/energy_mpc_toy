from typing import Dict

from omegaconf import DictConfig
from jaxtyping import Float
import torch
import torch.nn as nn

from models.actuators import LanguageActuator
from models.attention import Attention, TransformerBlock

from typing import Dict, Optional


class VariationalEncoder(nn.Module):
    """
    Perceiver encoder that downsamples inputs and outputs mean and log variance,
        per latent and per dimension (assumes diagonal covariance matrix).
    """

    def __init__(
        self,
        d_model: int,
        d_input: Optional[int] = None,
        vocab_size: Optional[int] = None,
        n_latents: int = 64,
        n_self_attn_layers: int = 4,
        n_self_attn_heads: int = 8,
        n_cross_attn_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        self.input_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
        )

        self.query = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)

        self.cross_attention = Attention(
            d_model=d_model,
            n_heads=n_cross_attn_heads,
            query_dim=d_model,
            key_dim=d_input if d_input is not None else d_model,
            value_dim=d_input,
        )

        self.pooling_norm = nn.RMSNorm(d_model)

        self.self_attn_layers = TransformerBlock(
            d_model=d_model,
            n_heads=n_self_attn_heads,
            n_layers=n_self_attn_layers,
            dropout=dropout,
        )

        self.final_norm = nn.RMSNorm(d_model)

        self.mu_projection = nn.Linear(d_model, d_model, bias=False)

        self.var_projection = nn.Linear(d_model, d_model, bias=False)

        # init
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            inputs_ids: [batch, seq_len]
            inputs_embeddings: [batch, seq_len, input_dim]
            attention_mask: [batch, seq_len]

        Returns: dictionary of:
            mean: [batch, n_latents, d_model]
            log_var: [batch, n_latents, d_model]
        """
        inputs = self.input_embedding(input_ids)

        batch_size, _, _ = inputs.shape

        query = self.query.expand(batch_size, -1, -1)

        latent = self.cross_attention(
            query=query,
            key=inputs,
            value=inputs,
            attn_mask=attention_mask,
        )

        latent = self.pooling_norm(latent)

        latent = self.self_attn_layers(latent)

        latent = self.final_norm(latent)

        mu = self.mu_projection(latent)

        log_var = self.var_projection(latent)

        return {
            "mean": mu,
            "log_var": log_var,
        }


class VariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        config: DictConfig,
        model_name: str,
        d_latent: int,
        n_latents: int = 256,
        n_self_attn_heads: int = 8,
        n_cross_attn_heads: int = 8,
        n_encoder_self_attn_layers: int = 6,
        n_decoder_self_attn_layers: int = 6,
        dropout: int = 0.1,
    ):

        super().__init__()

        self.config = config

        self.generative_model = LanguageActuator(
            model_name=model_name,
            latent_dim=d_latent,
            n_self_attn_layers=n_decoder_self_attn_layers,
            n_self_attn_heads=n_self_attn_heads,
            dropout=dropout,
        )

        vocab_size = self.generative_model.config.vocab_size

        self.recognition_model = VariationalEncoder(
            d_model=d_latent,
            vocab_size=vocab_size,
            n_latents=n_latents,
            n_self_attn_layers=n_encoder_self_attn_layers,
            n_self_attn_heads=n_self_attn_heads,
            n_cross_attn_heads=n_cross_attn_heads,
            dropout=dropout,
        )

    def from_pretrained(self, checkpoint) -> nn.Module:
        pass

    def forward(
        self,
        input_ids: Float[torch.Tensor, "batch seq_len"],
        attention_mask: Float[torch.Tensor, "batch seq_len"],
    ) -> Dict[str, Float[torch.Tensor, "batch seq_len dim"]]:

        variational_params = self.recognition_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        mu = variational_params["mean"]
        log_sigma = variational_params["log_var"]

        # Clamp to prevent compilation issues.
        mu = torch.clamp(mu, min=-10, max=10)
        log_sigma = torch.clamp(log_sigma, min=-10, max=10)

        # Reparameterization trick.
        epsilon = torch.randn_like(mu)
        sigma = (0.5 * log_sigma).exp()
        z = mu + sigma * epsilon

        logits = self.generative_model(
            z,
            target_tokens=input_ids,
            attention_mask=attention_mask,
        )

        return {
            "mean": mu,
            "log_var": log_sigma,
            "logits": logits,
        }
