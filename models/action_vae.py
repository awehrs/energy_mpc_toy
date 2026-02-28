from typing import Dict, Optional

import torch
import torch.nn as nn
from einops import repeat
from omegaconf import DictConfig
from jaxtyping import Int, Int32, Int64, Float
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding


from models.actuators import LanguageActuator
from models.attention import Attention, TransformerBlock


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

        self.rotary_emb = Qwen2RotaryEmbedding(
            config=Qwen2Config(
                hidden_size=d_model,
                num_attention_heads=n_cross_attn_heads,
            )
        )

        self.query = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)

        self.cross_attention = Attention(
            d_model=d_model,
            n_heads=n_cross_attn_heads,
            dropout=dropout,
            is_cross_attention=True,
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
        input_ids: Int[torch.Tensor, "batch seq_len"],
        max_seq_len: int,
        cu_seq_lens: Int32[torch.Tensor, "batch+1"],
        position_ids: Int64[torch.Tensor, "total_length"],
    ) -> Dict[str, Float[torch.Tensor, "n_latents latent dim"]]:

        inputs = self.input_embedding(input_ids)

        batch_size = len(cu_seq_lens) - 1
        max_seq_len_q = self.query.shape[0]
        cu_seq_lens_q = (
            torch.arange(
                len(cu_seq_lens),
                dtype=torch.int32,
                device=inputs.device,
            )
            * max_seq_len_q
        )
        query = repeat(self.query, "n_latents dim -> (b n_latents) dim", b=batch_size)
        if position_ids is not None:
            position_emb = self.rotary_emb(
                inputs.unsqueeze(0), position_ids.unsqueeze(0)
            )

        latent = self.cross_attention(
            query=query,
            key=inputs,
            position_emb=position_emb,
            max_seq_len_k=max_seq_len,
            max_seq_len_q=max_seq_len_q,
            cu_seq_lens_k=cu_seq_lens,
            cu_seq_lens_q=cu_seq_lens_q,
        )

        latent = self.pooling_norm(latent).to(torch.bfloat16)

        latent = self.self_attn_layers(
            q=latent,
            cu_seq_lens_q=cu_seq_lens_q,
            max_seq_len_q=max_seq_len_q,
        )

        latent = self.final_norm(latent).to(torch.bfloat16)

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
        n_latents: int = 64,
        vocab_size: int = 151665,
        pad_token_id: int = -100,
        max_action_len: int = 1024,
        n_self_attn_heads: int = 8,
        n_cross_attn_heads: int = 8,
        n_encoder_self_attn_layers: int = 6,
        n_decoder_self_attn_layers: int = 6,
        dropout: int = 0.1,
    ):

        super().__init__()

        self.config = config
        self.d_latent = d_latent
        self.n_latents = n_latents

        self.generative_model = LanguageActuator(
            model_name=model_name,
            n_latents=n_latents,
            latent_dim=d_latent,
            pad_token_id=pad_token_id,
            max_action_len=max_action_len,
            n_self_attn_layers=n_decoder_self_attn_layers,
            n_self_attn_heads=n_self_attn_heads,
            n_cross_attn_heads=n_cross_attn_heads,
            dropout=dropout,
        )

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
        corrupted_ids: Float[torch.Tensor, "batch seq_len"],
        max_seq_len: int,
        cu_seq_lens: Int32[torch.Tensor, "batch+1"],
        position_ids: Int64[torch.Tensor, "batch*seq_len"],
        targets: Optional[Float[torch.Tensor, "batch seq_len"]] = None,
        return_hidden_state: bool = False,
    ) -> Dict[str, Float[torch.Tensor, "batch seq_len dim"]]:

        variational_params = self.recognition_model(
            input_ids=input_ids,
            max_seq_len=max_seq_len,
            cu_seq_lens=cu_seq_lens,
            position_ids=position_ids,
        )

        mu = variational_params["mean"]
        log_sigma = variational_params["log_var"]

        # Clamp to prevent compilation issues.
        mu = torch.clamp(mu, min=-5, max=5)
        log_sigma = torch.clamp(log_sigma, min=-5, max=5)

        # Reparameterization trick.
        epsilon = torch.randn_like(mu)
        sigma = (0.5 * log_sigma).exp()
        z = mu + sigma * epsilon
        z = z.to(torch.bfloat16)

        outputs = self.generative_model(
            z,
            input_ids=corrupted_ids,
            position_ids=position_ids,
            max_seq_len_q=max_seq_len,
            cu_seq_lens_q=cu_seq_lens,
            targets=targets,
        )

        hidden_state = outputs["hidden_state"] if return_hidden_state else None
        loss = outputs["loss"]

        return {
            "hidden_state": hidden_state,
            "loss": loss,
            "mean": mu,
            "log_var": log_sigma,
        }
