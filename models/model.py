from typing import Dict

import einops
from omegaconf import DictConfig
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from models.decoder import Decoder
from models.encoder import Encoder
from models.energy_model import EnergyModel
from models.forward_model import ForwardModel
from models.action_projector import ActionProjection


class Model(nn.Module):

    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__()

        self.config = config

        pretrained_lm_name = config.pretrained_lm

        pretrained_config = AutoConfig.from_pretrained(pretrained_lm_name)

        assert pretrained_config.is_encoder_decoder == False

        pretrained_lm = AutoModelForCausalLM.from_pretrained(pretrained_lm_name)

        for param in pretrained_lm.parameters():
            param.requires_grad = False
        pretrained_lm.eval()

        # Derive hidden_dim
        if hasattr(pretrained_config, "hidden_size"):
            pretrained_hidden_dim = pretrained_config.hidden_size
        elif hasattr(self.config, "n_embed"):
            pretrained_hidden_dim = pretrained_config.n_embed
        elif hasattr(self.config, "d_model"):
            pretrained_hidden_dim = pretrained_config.d_model
        else:
            raise ValueError(
                "Could not determine pretrained_lm hidden dimension from config"
            )

        # Build encoder.
        self.encoder = Encoder(
            pretrained_llm=pretrained_lm,
            pretrained_llm_hidden_size=pretrained_hidden_dim,
            d_model=config.d_model,
            n_cross_attn_heads=config.n_cross_attn_heads,
            n_self_attn_heads=config.n_self_attn_heads,
            n_bottleneck_tokens=config.n_bottleneck_tokens,
            n_layers=config.n_encoder_layers,
            dropout=config.dropout,
        )

        # Build action projector.
        self.action_projection = ActionProjection(
            index_dim=config.index_dim,
            d_model=config.d_model,
            n_heads=config.n_cross_attn_heads,
            n_action_tokens=config.n_action_tokens,
            dropout=config.dropout,
        )

        # Build forward model.
        self.forward_model = ForwardModel(
            d_model=config.d_model,
            n_heads=config.n_self_attn_heads,
            n_layers=config.n_forward_layers,
            dropout=config.dropout,
            n_docs_per_step=config.n_docs,
            compressed_doc_len=config.n_bottleneck_tokens,
            n_action_tokens_per_step=config.n_action_tokens,
            max_steps=config.max_steps,
        )

        # Build decoder.
        self.decoder = Decoder(
            pretrained_decoder=pretrained_lm,
            pretrained_config=pretrained_config,
            latent_dim=config.d_model,
            n_heads=config.n_self_attn_heads,
            num_prefix_tokens=config.num_prefix_tokens,
            adapter_dim=config.adapter_dim,
            dropout=config.dropout,
            max_target_length=config.max_target_length,
        )

        # Build energy model.
        self.energy_model = EnergyModel(
            input_dim=config.d_model,
            d_model=config.d_energy_model,
            n_heads=config.n_cross_attn_heads,
            dropout=config.dropout,
        )

    def forward(
        self, input_ids: torch.Tensor, retrieval_queries: torch.FloatTensor, target_tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with teacher forcing (for training).
        
        Args:
            input_ids: [batch, n_retrieval_steps, n_docs_per_step, doc_len]
            retrieval_queries: [batch, n_retrieval_steps, n_action_tokens_per_step, index_dim]
            target_tokens: [batch, n_retrieval_steps, target_seq_len] - for teacher forcing

        Returns: Dict with keys -
            "document_latents": [batch, n_retrieval_steps, n_latent_tokens_per_step, d_model]
            "action_latents": [batch, n_retrieval_steps, n_action_tokens_per_step, d_model]
            "decoder_logits": [batch, n_retrieval_steps, target_seq_len, vocab_size]
            "energies": [batch, n_retrieval_steps]
        """
        bsz, steps, docs_per_step, seq_len = input_ids.shape

        # Embed retrieval queries to latent action tokens.
        retrieval_queries = einops.rearrange(retrieval_queries, "b s t d -> (b s) t d")

        action_latents = self.action_projection(
            retrieval_queries
        )  # [(b * n_retrieval_steps), n_action_tokens_per_step, d_model]

        action_latents = einops.rearrange(action_latents, "(b s) t d -> b s t d", b=bsz)

        # Encode / downsample documents.
        input_ids = einops.rearrange(input_ids, "b s d t -> (b s d) t")

        doc_latents = self.encoder(
            input_ids
        )  # [(batch * steps * n_docs), n_bottleneck_tokens, dim]

        doc_latents = einops.rearrange(
            doc_latents, "(b s d) n h -> b s (d n) h ", s=steps, d=docs_per_step
        )

        # Run through forward model.
        transformed_docs, transformed_acts = self.forward_model(
            doc_latents=doc_latents,
            action_latents=action_latents,
        )  # [batch, steps, n_{doc/action}_latents, dim]

        # Decode with teacher forcing.
        transformed_docs_flat = einops.rearrange(transformed_docs, " b s t d -> (b s) t d")
        target_tokens_flat = einops.rearrange(target_tokens, "b s t -> (b s) t")

        decoder_logits = self.decoder(latent=transformed_docs_flat, target_tokens=target_tokens_flat)

        decoder_logits = einops.rearrange(decoder_logits, "(b s) t v -> b s t v", b=bsz)

        # Calculate energy.
        energy = self.energy_model(transformed_docs_flat)

        energy = einops.rearrange(energy, "(b s) -> b s", b=bsz)

        return {
            "document_latents": transformed_docs,
            "action_latents": transformed_acts,
            "decoder_logits": decoder_logits,
            "energies": energy,
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        retrieval_queries: torch.FloatTensor,
        max_length: int = 512,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Generation mode (for inference).
        
        Args:
            input_ids: [batch, n_retrieval_steps, n_docs_per_step, doc_len]
            retrieval_queries: [batch, n_retrieval_steps, n_action_tokens_per_step, index_dim]
            max_length: Maximum generation length per step
            **kwargs: Additional generation arguments

        Returns: Dict with keys -
            "document_latents": [batch, n_retrieval_steps, n_latent_tokens_per_step, d_model]
            "action_latents": [batch, n_retrieval_steps, n_action_tokens_per_step, d_model]
            "generated_tokens": [batch, n_retrieval_steps, generated_seq_len]
            "energies": [batch, n_retrieval_steps]
        """
        bsz, steps, docs_per_step, seq_len = input_ids.shape

        # Embed retrieval queries to latent action tokens.
        retrieval_queries = einops.rearrange(retrieval_queries, "b s t d -> (b s) t d")

        action_latents = self.action_projection(
            retrieval_queries
        )  # [(b * n_retrieval_steps), n_action_tokens_per_step, d_model]

        action_latents = einops.rearrange(action_latents, "(b s) t d -> b s t d", b=bsz)

        # Encode / downsample documents.
        input_ids = einops.rearrange(input_ids, "b s d t -> (b s d) t")

        doc_latents = self.encoder(
            input_ids
        )  # [(batch * steps * n_docs), n_bottleneck_tokens, dim]

        doc_latents = einops.rearrange(
            doc_latents, "(b s d) n h -> b s (d n) h ", s=steps, d=docs_per_step
        )

        # Run through forward model.
        transformed_docs, transformed_acts = self.forward_model(
            doc_latents=doc_latents,
            action_latents=action_latents,
        )  # [batch, steps, n_{doc/action}_latents, dim]

        # Generate tokens.
        transformed_docs_flat = einops.rearrange(transformed_docs, " b s t d -> (b s) t d")

        generated_tokens = self.decoder.generate(latent=transformed_docs_flat, max_length=max_length, **kwargs)

        generated_tokens = einops.rearrange(generated_tokens, "(b s) t -> b s t", b=bsz)

        # Calculate energy.
        energy = self.energy_model(transformed_docs_flat)

        energy = einops.rearrange(energy, "(b s) -> b s", b=bsz)

        # Reshape outputs to match expected return shapes
        transformed_docs_reshaped = einops.rearrange(transformed_docs, "b s t d -> b s t d")
        
        return {
            "document_latents": transformed_docs_reshaped,
            "action_latents": transformed_acts,
            "generated_tokens": generated_tokens,
            "energies": energy,
        }
