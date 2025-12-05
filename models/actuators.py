from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from models.attention import TransformerBlock


class LanguageActuator(nn.Module):
    """
    Pre-trained causal LLM, wrapped with custom soft-prompting PEFT adapter.
    """

    def __init__(
        self,
        model_name: str,
        latent_dim: int = 768,
        n_self_attn_layers: int = 6,
        n_self_attn_heads: int = 8,
        dropout: int = 0.1,
    ):
        super().__init__()

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
            low_cpu_mem_usage=True,
            use_cache=False,
            # attn_implementation="flash_attention_2",
        )

        # self.decoder.gradient_checkpointing_enable()

        # Freeze all decoder parametersS
        for param in self.decoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        latent: torch.Tensor,
        target_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with target tokens (teacher forcing).

        Args:
            latent: [batch, n_latents, latent_dim] - encoded representation
            target_tokens: [batch, seq_len] - target sequence for teacher forcing
            attention_mask: Optional attention mask for target tokens

        Returns:
            logits: [batch, seq_len, vocab_size] - output logits
        """
        attn_out = self.self_attention_layers(latent)

        latent_embed = self.projection(attn_out)

        tokens_embed = self.decoder.get_input_embeddings()(target_tokens)

        combined_embed = torch.cat([latent_embed, tokens_embed], dim=1)

        batch_size, num_latent_tokens, _ = latent.shape

        latent_mask = torch.ones(
            batch_size,
            num_latent_tokens,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )

        combined_mask = torch.cat([latent_mask, attention_mask], dim=1)

        n_latents = latent.shape[1]

        return self.decoder(
            inputs_embeds=combined_embed, attention_mask=combined_mask
        ).logits[:, n_latents:, :]

    def generate(
        self,
        latent: torch.Tensor,
        max_length: int = 512,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 0.9,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text from latent representation.

        Args:
            latent: [batch, n_latents, latent_dim] - encoded representation
            max_length: Maximum generation length
            do_sample: Whether to sample or use greedy decoding
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            bos_token_id: Beginning-of-sequence token ID
            **kwargs: Additional generation arguments

        Returns:
            generated_ids: [batch, gen_len] - generated token IDs
        """
        # Set default token IDs if not provided
        if pad_token_id is None:
            pad_token_id = getattr(self.config, "pad_token_id", 0)
        if eos_token_id is None:
            eos_token_id = getattr(self.config, "eos_token_id", 1)

        bos_token_id = getattr(self.config, "bos_token_id", None)

        if bos_token_id is None:
            raise ValueError("Please provide bos_token_id for generation")

        kv_cache = self.adapter(latent)
        prefix_len = kv_cache.get_seq_length()
        bsz = latent.size(0)
        input_ids = torch.full(
            (bsz, 1), bos_token_id, device=latent.device, dtype=torch.long
        )
        cache_position = torch.arange(prefix_len, prefix_len + 1, device=latent.device)

        generated = self.pretrained_decoder.generate(
            input_ids=input_ids,
            past_key_values=kv_cache,
            cache_position=cache_position,
            use_cache=True,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        return generated

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
