import torch
import torch.nn as nn

from models.utils import Attention, TransformerBlock


class Encoder(nn.Module):
    """
    Encoder using cross-attention compression + transformer over sequence.
    """

    def __init__(
        self,
        pretrained_llm: nn.Module,
        pretrained_llm_hidden_size: int,
        d_model: int = 1024,
        n_cross_attn_heads: int = 8,
        n_self_attn_heads: int = 8,
        n_bottleneck_tokens: int = 16,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pretrained_llm = pretrained_llm
        self.d_model = d_model
        self.n_cross_attn_heads = n_cross_attn_heads
        self.n_self_attn_heads = n_self_attn_heads
        self.n_bottleneck_tokens = n_bottleneck_tokens
        self.n_layers = n_layers

        # Learnable compressed tokens
        self.bottleneck_tokens = nn.Parameter(
            torch.randn(1, self.n_bottleneck_tokens, d_model) * 0.02
        )

        # Cross-attention for compression: chunk features -> compressed representation
        self.compression_cross_attn = Attention(
            d_model=d_model,
            n_heads=n_cross_attn_heads,
            query_dim=d_model,
            key_dim=pretrained_llm_hidden_size,
            value_dim=pretrained_llm_hidden_size,
            output_dim=d_model,
            dropout=dropout,
        )

        # Layer norm after compression
        self.compression_norm = nn.LayerNorm(d_model)

        # Transformer for temporal modeling over compressed representations
        self.self_attention_transformer = TransformerBlock(
            d_model=d_model,
            n_heads=n_self_attn_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Pretrained encoder pre-processing -> Cross-attention compression -> Self-attention post-processing.

        Args:
            input_ids: [batch, seq_len]
            n_docs: number of documents being encoded.

        Returns:
            latent: [batch, n_bottleneck_tokens, d_model]
        """
        batch_size = input_ids.shape[0]

        # expand learnable bottleneck tokens
        bottleneck_tokens = self.bottleneck_tokens.expand(
            batch_size, -1, -1
        )  # [batch, n_bottleneck_tokens, d_model]

        # run through pretrained LLM
        outputs = self.pretrained_llm(input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True)
        # Handle both CausalLM and base transformer outputs
        if hasattr(outputs, 'last_hidden_state'):
            preprocessed_inputs = outputs.last_hidden_state  # [batch, t, hidden_size]
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            preprocessed_inputs = outputs.hidden_states[-1]  # [batch, t, hidden_size]
        else:
            # Fallback: use logits (not ideal but prevents crash)
            raise ValueError("Could not extract hidden states from pretrained model output")

        # compress with cross-attention
        compressed = self.compression_cross_attn(
            query=bottleneck_tokens, key=preprocessed_inputs, value=preprocessed_inputs
        )  # [batch, n_bottleneck_tokens, d_model]

        compressed = self.compression_norm(compressed)

        # temporal self-attention
        processed = self.self_attention_transformer(compressed)

        return processed
