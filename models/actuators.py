from typing import Optional

import torch
import torch.nn as nn
from jaxtyping import Float
from transformers import AutoConfig, AutoModelForCausalLM

from models.attention import TransformerBlock

import logging


class LanguageActuator(nn.Module):
    """
    Pre-trained causal LLM, wrapped with custom soft-prompting PEFT adapter.
    """

    def __init__(
        self,
        model_name: str,
        n_latents: int = 64,
        latent_dim: int = 768,
        vocab_size: int = 151666,
        pad_token_id: int = -100,
        max_action_len: int = 1024,
        n_self_attn_layers: int = 6,
        n_self_attn_heads: int = 8,
        dropout: int = 0.1,
    ):
        super().__init__()

        self.n_latents = n_latents
        self.pad_token_id = pad_token_id
        self.max_action_len = max_action_len

        # Extract decoder configuration
        self.config = AutoConfig.from_pretrained(model_name)

        # Get decoder hidden dimension
        if hasattr(self.config, "hidden_size"):
            decoder_hidden_dim = self.config.hidden_size
        elif hasattr(self.config, "d_model"):
            decoder_hidden_dim = self.config.d_model
        elif hasattr(self.config, "n_embed"):
            decoder_hidden_dim = self.config.n_embed
        else:
            raise ValueError("Could not determine decoder hidden dimension from config")

        self.self_attention_layers = TransformerBlock(
            d_model=latent_dim,
            n_heads=n_self_attn_heads,
            n_layers=n_self_attn_layers,
            dropout=dropout,
        )

        self.projection = nn.Linear(latent_dim, decoder_hidden_dim, bias=False)

        self.decoder = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        self.decoder.gradient_checkpointing_enable()

        # Freeze all decoder parameters except for resized token embedding.
        for param in self.decoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        latent: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        latent_indices: Optional[torch.Tensor] = None,
        padding_indices: Optional[torch.Tensor] = None,
    ) -> Float[torch.Tensor, "batch seq_len vocab_size"]:

        if len(latent.shape) == 2:
            # Packed sequence
            total_latents = latent.shape[0]
            batch_size = total_latents // self.n_latents

            cu_seq_lens_q = (
                torch.arange(batch_size + 1, dtype=torch.int32, device=latent.device)
                * self.n_latents
            )

            max_seq_len_q = self.n_latents

            attn_out = self.self_attention_layers(
                q=latent,
                cu_seq_lens_q=cu_seq_lens_q,
                max_seq_len_q=max_seq_len_q,
            )

            latent_embed = self.projection(attn_out)

            total_positions = position_ids.shape[0]

            dim = latent_embed.shape[-1]

            combined = torch.empty(
                (total_positions, dim),
                dtype=latent_embed.dtype,
                device=latent_embed.device,
            )

            combined[latent_indices] = latent_embed

            padding_embeds = self.decoder.get_input_embeddings()(
                torch.full(
                    (len(padding_indices),),
                    self.pad_token_id,
                    dtype=torch.long,
                    device=latent.device,
                )
            )
            combined[padding_indices] = padding_embeds

            logits = self.decoder(
                inputs_embeds=combined.unsqueeze(0),
                position_ids=position_ids.unsqueeze(0),
            ).logits.squeeze(0)

            logits = logits[padding_indices]
        else:
            # Batched sequence
            attn_out = self.self_attention_layers(q=latent)

            latent_embed = self.projection(attn_out)

            batch_size, _, dim = latent_embed.shape

            padding = torch.full(
                size=(batch_size, self.max_action_len, dim),
                fill_value=self.pad_token_id,
                device=latent_embed.device,
            )

            latent_embed = torch.cat([latent_embed, padding], dim=1)

            logits = self.decoder(
                inputs_embeds=latent_embed,
            ).logits[:, self.n_latents :, :]

        return logits

    def generate(
        self,
        latent: torch.Tensor,
        max_length: int = 512,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.9,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate sequences from latent representation.

        Args:
            latent: [batch, n_latents, d_latent] - encoded representation

        Returns:
            generated_ids: [batch, seq_len] - generated token IDs
        """
        if len(latent.shape) == 2:
            # Packed sequence
            total_latents = latent.shape[0]
            batch_size = total_latents // self.n_latents

            cu_seq_lens_q = (
                torch.arange(batch_size + 1, dtype=torch.int32, device=latent.device)
                * self.n_latents
            )

            max_seq_len_q = self.n_latents

            attn_out = self.self_attention_layers(
                q=latent,
                cu_seq_lens_q=cu_seq_lens_q,
                max_seq_len_q=max_seq_len_q,
            )

            latent_embed = self.projection(attn_out)

            latent_embed = latent_embed.view(batch_size, self.n_latents, -1)

        else:
            # Batched sequence
            attn_out = self.self_attention_layers(q=latent)

            latent_embed = self.projection(attn_out)

        outputs = self.decoder.generate(
            inputs_embeds=latent_embed,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        return outputs

    def get_num_parameters(self) -> dict:
        """Get parameter counts for analysis."""
        total_params = sum(p.numel() for p in self.parameters())
        adapter_params = sum(p.numel() for p in self.adapter.parameters())
        decoder_params = sum(p.numel() for p in self.pretrained_decoder.parameters())

        return {
            "total_parameters": total_params,
            "adapter_parameters": adapter_params,
            "pretrained_decoder_parameters": decoder_params,
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }


if __name__ == "__main__":
    LanguageActuator(model_name="gpt2")
