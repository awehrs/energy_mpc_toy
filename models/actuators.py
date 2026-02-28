from typing import Dict, Optional, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils.generic import TransformersKwargs
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2Attention,
    Qwen2PreTrainedModel,
    Qwen2RotaryEmbedding,
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from models.attention import Attention, TransformerBlock


class Actuator:
    pass


class LatentCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.attn = Attention(
            d_model=hidden_size,
            n_heads=n_heads,
            is_causal=False,
            is_cross_attention=True,
        )
        self.norm_q = Qwen2RMSNorm(hidden_size, eps)
        self.norm_kv = Qwen2RMSNorm(hidden_size, eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        latents: torch.Tensor,
        max_seq_len_q: int,
        max_seq_len_k: int,
        cu_seq_lens_q: torch.Tensor,
        cu_seq_lens_k: torch.Tensor,
    ) -> torch.Tensor:

        bsz = hidden_states.shape[0]
        assert bsz == 1
        assert len(cu_seq_lens_q.shape) == len(cu_seq_lens_k.shape) == 1

        q = self.norm_q(hidden_states)
        q = einops.rearrange(q, "b t h -> (b t) h")

        kv = self.norm_kv(latents)

        h = self.attn(
            q,
            kv,
            max_seq_len_q=max_seq_len_q,
            max_seq_len_k=max_seq_len_k,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
        )

        h = einops.rearrange(h, "(b t) h -> b t h", b=bsz)

        return h


class Qwen2DecoderLayerWithLatents(nn.Module):
    gradient_checkpointing = False

    def __init__(
        self,
        config: Qwen2Config,
        layer_idx: int,
        n_cross_attn_heads: int = 8,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)

        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.attention_type = config.layer_types[layer_idx]

        self.latent_xattn = LatentCrossAttention(
            hidden_size=config.hidden_size,
            n_heads=n_cross_attn_heads,
            eps=config.rms_norm_eps,
        )

        self.latent_xattn_norm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        latents: torch.Tensor,
        max_seq_len_q: int,
        max_seq_len_k: int,
        cu_seq_lens_q: torch.Tensor,
        cu_seq_lens_k: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        # Original self-attn block.
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Cross-attn to latents.
        x = self.latent_xattn_norm(hidden_states)

        hidden_states = hidden_states + self.latent_xattn(
            x,
            latents=latents,
            max_seq_len_q=max_seq_len_q,
            max_seq_len_k=max_seq_len_k,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
        )

        # Original MLP block.
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen2ModelWithLatents(Qwen2PreTrainedModel):
    gradient_checkpointing = False

    def __init__(
        self,
        config: Qwen2Config,
        n_cross_attn_heads: int = 8,
    ):
        super().__init__(config)

        assert config.hidden_size % n_cross_attn_heads == 0

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayerWithLatents(
                    config, layer_idx=i, n_cross_attn_heads=n_cross_attn_heads
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.post_init()

    def forward(
        self,
        latents: torch.Tensor,
        max_seq_len_q: int,
        max_seq_len_k: int,
        cu_seq_lens_q: torch.Tensor,
        cu_seq_lens_k: torch.Tensor,
        position_ids: torch.LongTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen,
                past_seen + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # prepare causal mask mapping (copied from HF)
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(**mask_kwargs)
                )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    layer,
                    hidden_states,
                    latents,
                    max_seq_len_q,
                    max_seq_len_k,
                    cu_seq_lens_q,
                    cu_seq_lens_k,
                    position_ids,
                    causal_mask_mapping[layer.attention_type],
                    past_key_values,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    latents=latents,
                    max_seq_len_q=max_seq_len_q,
                    max_seq_len_k=max_seq_len_k,
                    cu_seq_lens_q=cu_seq_lens_q,
                    cu_seq_lens_k=cu_seq_lens_k,
                    position_ids=position_ids,
                    attention_mask=causal_mask_mapping[layer.attention_type],
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Qwen2ForCausalLMWithLatents(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    gradient_checkpointing = False

    def __init__(
        self,
        config: Qwen2Config,
        n_cross_attn_heads: int = 8,
    ):
        super().__init__(config)
        self.model = Qwen2ModelWithLatents(
            config,
            n_cross_attn_heads=n_cross_attn_heads,
        )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        latents: torch.Tensor,
        max_seq_len_q: int,
        max_seq_len_k: int,
        cu_seq_lens_q: torch.Tensor,
        cu_seq_lens_k: torch.Tensor,
        position_ids: torch.LongTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:

        outputs: BaseModelOutputWithPast = self.model(
            latents=latents,
            max_seq_len_q=max_seq_len_q,
            max_seq_len_k=max_seq_len_k,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            position_ids=position_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def load_qwen_with_latents(
    model_name: str,
    dtype=torch.bfloat16,
    attn_impl: str = "flash_attention_2",
    n_cross_attn_heads: int = 8,
) -> nn.Module:
    # Load HF model (source of truth weights)
    hf = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation=attn_impl,
    )
    cfg = hf.config

    # Build forked model
    fork = Qwen2ForCausalLMWithLatents(
        cfg,
        n_cross_attn_heads=n_cross_attn_heads,
    ).to(dtype)

    # Copy base model weights.
    missing, unexpected = fork.model.load_state_dict(
        hf.model.state_dict(), strict=False
    )

    # Copy lm_head weights
    fork.lm_head.load_state_dict(hf.lm_head.state_dict(), strict=True)

    # Sanity: only new latent_xattn weights should be missing
    print(
        "Missing keys (expected mostly latent_xattn):", missing[:10], "…", len(missing)
    )
    print("Unexpected keys:", unexpected[:10], "…", len(unexpected))

    # Freeze everything except latent cross-attn modules.
    for n, p in fork.named_parameters():
        p.requires_grad = ("latent_xattn" in n) or ("latent_xattn_norm" in n)

    return fork


class LanguageActuator(nn.Module):
    """
    Pre-trained causal LLM, wrapped with custom soft-prompting PEFT adapter.
    """

    def __init__(
        self,
        model_name: str,
        n_latents: int = 64,
        latent_dim: int = 768,
        pad_token_id: int = -100,
        max_action_len: int = 1024,
        n_self_attn_layers: int = 6,
        n_self_attn_heads: int = 8,
        n_cross_attn_heads: int = 8,
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

        self.decoder = load_qwen_with_latents(
            model_name,
            dtype=torch.bfloat16,
            attn_impl="flash_attention_2",
            n_cross_attn_heads=n_cross_attn_heads,
        )

        # self.decoder.gradient_checkpointing_enable()

    def forward(
        self,
        latents: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        max_seq_len_q: int,
        cu_seq_lens_q: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        assert len(latents.shape) == 2
        assert len(input_ids.shape) == 1
        assert len(position_ids.shape) == 1
        assert len(cu_seq_lens_q.shape) == 1
        if targets is not None:
            assert len(targets.shape) == 1

        total_latents = latents.shape[0]
        batch_size = total_latents // self.n_latents

        cu_seq_lens_latents = (
            torch.arange(batch_size + 1, dtype=torch.int32, device=latents.device)
            * self.n_latents
        )

        max_seq_len_latents = self.n_latents

        attn_out = self.self_attention_layers(
            q=latents,
            cu_seq_lens_q=cu_seq_lens_latents,
            max_seq_len_q=max_seq_len_latents,
        )

        latent_embed = self.projection(attn_out)

        input_embeds = self.decoder.model.embed_tokens(input_ids)

        input_embeds = einops.rearrange(input_embeds, "t h -> 1 t h")

        position_ids = einops.rearrange(position_ids, "t -> 1 t")

        hidden_states = self.decoder.model(
            latents=latent_embed,
            max_seq_len_q=max_seq_len_q,
            max_seq_len_k=max_seq_len_latents,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_latents,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
        ).last_hidden_state.squeeze(0)

        if targets is not None:
            total_loss = hidden_states.new_zeros(())
            total_tokens = 0
            chunk_size = 512

            for i in range(0, hidden_states.size(0), chunk_size):
                chunk_hidden = hidden_states[i : i + chunk_size]
                chunk_targets = targets[i : i + chunk_size]

                chunk_logits = self.decoder.lm_head(chunk_hidden)
                chunk_loss = F.cross_entropy(
                    chunk_logits,
                    chunk_targets,
                    ignore_index=-100,
                    reduction="sum",
                )

                total_loss = total_loss + chunk_loss
                total_tokens += (chunk_targets != -100).sum().item()

            loss = total_loss / max(total_tokens, 1)

        else:
            loss = None

        return {
            "hidden_states": hidden_states,
            "loss": loss,
        }

    @torch.no_grad()
    def generate(
        self,
        latent: torch.Tensor,
        max_length: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.9,
        bos_token_id: int = 1,
    ) -> torch.Tensor:
        """
        Generate sequences from latent representations (batched).

        Args:
            latent: [batch * n_latents, d_latent] - packed latents

        Returns:
            generated_ids: [batch, max_length] - generated token IDs
        """
        device = latent.device
        total_latents = latent.shape[0]
        batch_size = total_latents // self.n_latents

        # Process latents through self-attention and projection
        cu_seq_lens_latents = (
            torch.arange(batch_size + 1, dtype=torch.int32, device=device)
            * self.n_latents
        )
        max_seq_len_latents = self.n_latents

        attn_out = self.self_attention_layers(
            q=latent,
            cu_seq_lens_q=cu_seq_lens_latents,
            max_seq_len_q=max_seq_len_latents,
        )
        latent_embed = self.projection(attn_out)

        # Initialize with BOS token for each sequence
        generated = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long, device=device
        )

        for step in range(max_length - 1):
            seq_len = generated.shape[1]

            # Pack sequences: [batch, seq] -> [batch * seq]
            input_ids = generated.reshape(-1)

            # Position IDs: [0, 1, ..., seq_len-1] repeated for each batch
            position_ids = (
                torch.arange(seq_len, dtype=torch.long, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1)
                .reshape(-1)
            )

            # cu_seq_lens for packed sequences
            cu_seq_lens_q = (
                torch.arange(batch_size + 1, dtype=torch.int32, device=device) * seq_len
            )

            # Embed and reshape for decoder
            input_embeds = self.decoder.model.embed_tokens(input_ids)
            input_embeds = input_embeds.unsqueeze(0)  # [1, batch*seq, hidden]
            position_ids = position_ids.unsqueeze(0)  # [1, batch*seq]

            # Forward through decoder
            hidden_states = self.decoder.model(
                latents=latent_embed,
                max_seq_len_q=seq_len,
                max_seq_len_k=max_seq_len_latents,
                cu_seq_lens_q=cu_seq_lens_q,
                cu_seq_lens_k=cu_seq_lens_latents,
                position_ids=position_ids,
                inputs_embeds=input_embeds,
            ).last_hidden_state.squeeze(0)

            # Get last token hidden state for each sequence
            last_indices = cu_seq_lens_q[1:] - 1  # Last position of each sequence
            last_hidden = hidden_states[last_indices]  # [batch, hidden]

            # Get logits
            logits = self.decoder.lm_head(last_hidden)  # [batch, vocab]

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Apply top-p sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample next tokens
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)  # [batch, 1]

            # Append to generated
            generated = torch.cat([generated, next_tokens], dim=1)

        return generated

    def get_num_parameters(self) -> dict:
        """Get parameter counts for analysis."""
        total_params = sum(p.numel() for p in self.parameters())
        self_attn_params = sum(
            p.numel() for p in self.self_attention_layers.parameters()
        )
        projection_params = sum(p.numel() for p in self.projection.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())

        return {
            "total_parameters": total_params,
            "self_attention_parameters": self_attn_params,
            "projection_parameters": projection_params,
            "decoder_parameters": decoder_params,
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }


if __name__ == "__main__":
    LanguageActuator(model_name="gpt2")
