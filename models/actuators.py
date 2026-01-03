from typing import Optional

import torch
import torch.nn as nn
from jaxtyping import Int64, Float
from transformers import AutoConfig, AutoModelForCausalLM

from models.attention import TransformerBlock


class LanguageActuator(nn.Module):
    """
    Pre-trained causal LLM, wrapped with custom soft-prompting PEFT adapter.
    """

    def __init__(
        self,
        model_name: str,
        n_latents: int = 64,
        latent_dim: int = 768,
        n_self_attn_layers: int = 6,
        n_self_attn_heads: int = 8,
        dropout: int = 0.1,
    ):
        super().__init__()

        self.n_latents = n_latents

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

        # Freeze all decoder parameters
        for param in self.decoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        latent: torch.Tensor,
        target_tokens: torch.Tensor,
        token_indices: Optional[Int64[torch.Tensor, "total_batch_tokens"]] = None,
        latent_indices: Optional[Int64[torch.Tensor, "total_batch_latents"]] = None,
        position_ids: Optional[
            Int64[torch.Tensor, "total_batch_tokens_and_latents"]
        ] = None,
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
            tokens_embed = self.decoder.get_input_embeddings()(target_tokens)

            total_len = len(latent_indices) + len(token_indices)
            dim = tokens_embed.shape[-1]
            combined_embed = torch.zeros(
                total_len, dim, device=tokens_embed.device, dtype=tokens_embed.dtype
            )
            combined_embed.index_copy_(0, latent_indices, latent_embed)
            combined_embed.index_copy_(0, token_indices, tokens_embed)

            logits = self.decoder(
                inputs_embeds=combined_embed.unsqueeze(0),
                position_ids=position_ids.unsqueeze(0),
            ).logits.squeeze(0)[token_indices]

        else:
            # Batched sequence
            attn_out = self.self_attention_layers(q=latent)

            latent_embed = self.projection(attn_out)
            tokens_embed = self.decoder.get_input_embeddings()(target_tokens)

            combined_embed = torch.cat([latent_embed, tokens_embed], dim=1)
            n_latents = latent.shape[1]

            logits = self.decoder(
                inputs_embeds=combined_embed,
            ).logits[:, n_latents:, :]

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
