import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Model
import os

from models.action_projector import ActionProjection
from models.decoder import Decoder
from models.encoder import Encoder
from models.energy_model import EnergyModel
from models.forward_model import ForwardModel
from models.model import Model
from models.query_embedder import QueryEmbedder


@pytest.mark.parametrize("batch_size,latent_len,latent_dim", [(2, 10, 256)])
def test_prefix_adapter_with_gpt2_cpu(batch_size, latent_len, latent_dim):
    device = torch.device("cpu")

    # Load GPT-2 config and model (on CPU)
    cfg = GPT2Config.from_pretrained("gpt2")
    base = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    # Wrap with our adapter
    wrapper = Decoder(
        pretrained_decoder=base,
        pretrained_config=cfg,
        latent_dim=latent_dim,
        num_prefix_tokens=8,
    ).to(device)

    # Fake latent input
    latent = torch.randn(batch_size, latent_len, latent_dim, device=device)

    # Fake target tokens
    tgt_len = 6
    targets = torch.randint(
        low=0, high=cfg.vocab_size, size=(batch_size, tgt_len), device=device
    )

    # Forward pass (teacher forcing)
    logits = wrapper(latent=latent, target_tokens=targets)
    assert logits.shape == (batch_size, tgt_len, cfg.vocab_size)

    # Check grads flow
    loss = logits.mean()
    loss.backward()
    grad_norm = sum(
        p.grad.abs().sum().item() for p in wrapper.parameters() if p.grad is not None
    )
    assert grad_norm > 0

    # Generation test
    gen_ids = wrapper.generate(latent=latent, max_length=12, do_sample=False)
    assert gen_ids.dim() == 2  # [B, seq_len]
    assert gen_ids.size(0) == batch_size
    assert gen_ids.size(1) <= 12


@pytest.mark.parametrize("batch_size,seq_len", [(1, 40), (2, 32)])
def test_encoder_with_gpt2(batch_size, seq_len):
    device = torch.device("cpu")

    # tiny GPT-2 for speed
    cfg = GPT2Config.from_pretrained("gpt2")
    base_model = GPT2Model.from_pretrained("gpt2").to(device)

    encoder = Encoder(
        pretrained_llm=base_model,
        pretrained_llm_hidden_size=cfg.hidden_size,
        d_model=256,
        n_cross_attn_heads=4,
        n_self_attn_heads=4,
        n_bottleneck_tokens=8,
        n_layers=2,
        dropout=0.1,
    ).to(device)

    # input_ids shaped [batch, seq_len] - one document per batch item
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)

    latent = encoder(input_ids)

    # latent shape should be [batch, n_bottleneck_tokens, d_model]
    assert latent.shape == (
        batch_size,
        encoder.n_bottleneck_tokens,
        encoder.d_model,
    )

    # forward + backward works
    loss = latent.mean()
    loss.backward()
    grad_norm = sum(
        p.grad.abs().sum().item() for p in encoder.parameters() if p.grad is not None
    )
    assert grad_norm > 0


@pytest.mark.parametrize("batch_size,n_steps", [(2, 2), (1, 3)])
def test_forward_model_forward_pass(batch_size, n_steps):
    # Test config
    d_model = 32
    n_docs_per_step = 2
    compressed_doc_len = 4
    n_action_tokens_per_step = 1

    model = ForwardModel(
        d_model=d_model,
        n_heads=4,
        n_layers=2,
        n_docs_per_step=n_docs_per_step,
        compressed_doc_len=compressed_doc_len,
        n_action_tokens_per_step=n_action_tokens_per_step,
        max_steps=n_steps,
    )

    # Create input tensors
    doc_latents = torch.randn(
        batch_size,
        n_steps,
        n_docs_per_step * compressed_doc_len,
        d_model,
        requires_grad=True,
    )
    action_latents = torch.randn(
        batch_size, n_steps, n_action_tokens_per_step, d_model, requires_grad=True
    )

    # Forward pass
    docs_out, acts_out = model(doc_latents, action_latents)

    # Shape validation
    expected_doc_shape = (
        batch_size,
        n_steps,
        n_docs_per_step * compressed_doc_len,
        d_model,
    )
    expected_act_shape = (batch_size, n_steps, n_action_tokens_per_step, d_model)

    assert (
        docs_out.shape == expected_doc_shape
    ), f"Doc output shape: got {docs_out.shape}, expected {expected_doc_shape}"
    assert (
        acts_out.shape == expected_act_shape
    ), f"Action output shape: got {acts_out.shape}, expected {expected_act_shape}"

    # Gradient test
    loss = docs_out.sum() + acts_out.sum()
    loss.backward()

    # Check gradients flow to inputs
    assert doc_latents.grad is not None, "Doc latents should have gradients"
    assert action_latents.grad is not None, "Action latents should have gradients"

    # Check gradients flow to model parameters
    grad_norm = sum(
        p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None
    )
    assert grad_norm > 0, "Model parameters should have gradients"


def test_forward_model_mask_rules_cpu():
    # Small config for testing
    D = 2  # doc tokens per step
    A = 1  # action tokens per step
    max_steps = 3
    model = ForwardModel(
        d_model=16,
        n_heads=2,
        n_layers=1,
        n_docs_per_step=1,
        compressed_doc_len=D,
        n_action_tokens_per_step=A,
    )
    model.max_steps = max_steps
    model._setup_flex_attention()  # rebuild mask

    block_mask = model.block_mask_full
    Q_LEN = KV_LEN = max_steps * (D + A)

    # Convert BlockMask into dense boolean tensor [Q_LEN, KV_LEN]
    mask_dense = block_mask.to_dense().squeeze(0).squeeze(0)  # [Q_LEN, KV_LEN]

    for q_idx in range(Q_LEN):
        q_step, q_off = divmod(q_idx, D + A)
        is_q_doc = q_off < D
        is_q_action = q_off >= D

        for kv_idx in range(KV_LEN):
            kv_step, kv_off = divmod(kv_idx, D + A)
            is_kv_doc = kv_off < D
            same_step = q_step == kv_step

            allowed = mask_dense[q_idx, kv_idx].item()

            if kv_step < q_step:
                # All queries can attend to all tokens from previous steps
                assert allowed, f"q={q_idx}, kv={kv_idx} should see prev step"
            elif same_step:
                if is_q_doc:
                    # Doc tokens can attend to all doc tokens in same step
                    # Doc tokens CANNOT attend to action tokens in same step
                    expected = is_kv_doc
                    assert (
                        allowed == expected
                    ), f"q={q_idx}, kv={kv_idx} doc same-step: got {allowed}, expected {expected}"
                elif is_q_action:
                    # Action tokens can attend to all tokens (doc + action) in same step
                    assert (
                        allowed
                    ), f"q={q_idx}, kv={kv_idx} action should see all same-step"
            else:
                # q_step < kv_step → future → never allowed (causal constraint)
                assert not allowed, f"q={q_idx}, kv={kv_idx} shouldn't see future"


@pytest.mark.parametrize("batch_size, n_latents, d_model", [(2, 16, 32), (4, 8, 64)])
def test_energy_model_forward(batch_size, n_latents, d_model):
    # Create dummy latent input
    latent = torch.randn(batch_size, n_latents, d_model, requires_grad=True)

    # Build model (input_dim = d_model for this test)
    model = EnergyModel(
        input_dim=d_model,
        d_model=d_model,
        n_heads=4,
    )

    # Forward pass
    energy = model(latent)

    # Check output shape
    assert energy.shape == (
        batch_size,
    ), f"Expected shape ({batch_size},), got {energy.shape}"

    # Check that output requires grad
    assert energy.requires_grad, "Energy output should require gradients"

    # Backward pass works
    energy.sum().backward()
    assert latent.grad is not None, "Latent input should have gradients after backward"

    # Check that changing latent changes energy
    energy_1 = model(latent.detach())
    energy_2 = model(latent.detach() + 0.1)
    assert not torch.allclose(
        energy_1, energy_2
    ), "Energy should change when latent changes"


def test_query_embedder_forward():
    batch_size = 4
    n_latents = 10
    input_dim = 32
    index_dim = 64

    model = QueryEmbedder(
        input_dim=input_dim,
        index_dim=index_dim,
        d_model=128,
        n_heads=4,
    )
    latent = torch.randn(batch_size, n_latents, input_dim)

    # Forward pass
    out = model(latent)

    # Shape check
    assert out.shape == (batch_size, index_dim)

    # Normalization check: each row should have norm ~1
    norms = out.norm(dim=-1)
    assert torch.allclose(
        norms, torch.ones_like(norms), atol=1e-5
    ), f"Norms not close to 1: {norms}"

    # Gradient check
    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


@pytest.mark.parametrize("batch_size,n_latent_action_tokens", [(2, 3), (4, 5)])
def test_action_projection_forward(batch_size, n_latent_action_tokens):
    # Test config: queries have 1 action token per batch, project to n>1 latent action tokens
    n_query_action_tokens = 1  # One action token per batch (as specified)
    index_dim = 64
    d_model = 128
    n_heads = 4

    model = ActionProjection(
        index_dim=index_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_action_tokens=n_latent_action_tokens,  # n>1 latent action tokens
        dropout=0.1,
    )

    # Input queries: [batch, n_query_action_tokens=1, index_dim]
    queries = torch.randn(
        batch_size, n_query_action_tokens, index_dim, requires_grad=True
    )

    # Forward pass
    latent_actions = model(queries)

    # Shape validation: should output [batch, n_latent_action_tokens, d_model]
    expected_shape = (batch_size, n_latent_action_tokens, d_model)
    assert (
        latent_actions.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {latent_actions.shape}"

    # Gradient flow check
    loss = latent_actions.sum()
    loss.backward()

    # Check gradients flow to input queries
    assert queries.grad is not None, "Input queries should have gradients"
    assert (
        queries.grad.shape == queries.shape
    ), "Query gradients should match input shape"

    # Check gradients flow to model parameters
    grad_norm = sum(
        p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None
    )
    assert grad_norm > 0, "Model parameters should have gradients"

    # Test that different inputs produce different outputs
    queries2 = torch.randn(batch_size, n_query_action_tokens, index_dim)
    latent_actions2 = model(queries2)
    assert not torch.allclose(
        latent_actions.detach(), latent_actions2
    ), "Different inputs should produce different outputs"

    # Test that action_query parameter has correct shape
    assert model.action_query.shape == (
        n_latent_action_tokens,
        d_model,
    ), f"action_query should have shape ({n_latent_action_tokens}, {d_model})"


@pytest.mark.parametrize("batch_size,n_steps", [(1, 2), (2, 3)])
def test_main_model_forward(batch_size, n_steps):
    # Load debug config with Hydra
    config_dir = os.path.join(os.path.dirname(__file__), "../conf")
    
    with initialize_config_dir(config_dir=os.path.abspath(config_dir), version_base="1.1"):
        config = compose(config_name="config", overrides=["model=debug", f"max_steps={n_steps}"])
    
    # Build model
    model = Model(config.model)
    
    # Create test inputs matching expected shapes
    n_docs_per_step = config.n_docs
    doc_len = 20
    n_action_tokens_per_step = config.model.n_action_tokens
    index_dim = config.index_dim
    
    # Input shapes: [batch, n_retrieval_steps, n_docs_per_step, doc_len]
    input_ids = torch.randint(
        0, 1000,  # Use smaller vocab for testing
        (batch_size, n_steps, n_docs_per_step, doc_len),
        requires_grad=False
    )
    
    # Retrieval queries: [batch, n_retrieval_steps, n_action_tokens_per_step, index_dim]
    retrieval_queries = torch.randn(
        batch_size, n_steps, n_action_tokens_per_step, index_dim,
        requires_grad=True
    )
    
    # Target tokens for teacher forcing: [batch, n_steps, target_seq_len]
    target_seq_len = 10
    target_tokens = torch.randint(
        0, 1000,  # Use smaller vocab for testing
        (batch_size, n_steps, target_seq_len),
        requires_grad=False
    )
    
    # Forward pass
    outputs = model(input_ids, retrieval_queries, target_tokens)
    
    # Shape validation
    assert isinstance(outputs, dict), "Model should return a dictionary"
    expected_keys = {"document_latents", "action_latents", "decoder_logits", "energies"}
    assert set(outputs.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(outputs.keys())}"
    
    # Check output shapes
    n_bottleneck_per_step = n_docs_per_step * config.model.n_bottleneck_tokens
    d_model = config.model.d_model
    vocab_size = model.decoder.vocab_size
    
    # document_latents: [batch, n_steps, n_latent_tokens_per_step, d_model]
    expected_doc_shape = (batch_size, n_steps, n_bottleneck_per_step, d_model)
    assert outputs["document_latents"].shape == expected_doc_shape, \
        f"document_latents shape: got {outputs['document_latents'].shape}, expected {expected_doc_shape}"
    
    # action_latents: [batch, n_steps, n_action_tokens_per_step, d_model]
    expected_action_shape = (batch_size, n_steps, n_action_tokens_per_step, d_model)
    assert outputs["action_latents"].shape == expected_action_shape, \
        f"action_latents shape: got {outputs['action_latents'].shape}, expected {expected_action_shape}"
    
    # decoder_logits: [batch, n_steps, target_seq_len, vocab_size]
    expected_logits_shape = (batch_size, n_steps, target_seq_len, vocab_size)
    assert outputs["decoder_logits"].shape == expected_logits_shape, \
        f"decoder_logits shape: got {outputs['decoder_logits'].shape}, expected {expected_logits_shape}"
    
    # energies: [batch, n_steps]
    expected_energy_shape = (batch_size, n_steps)
    assert outputs["energies"].shape == expected_energy_shape, \
        f"energies shape: got {outputs['energies'].shape}, expected {expected_energy_shape}"
    
    # Gradient flow test
    total_loss = (
        outputs["document_latents"].sum() + 
        outputs["action_latents"].sum() +
        outputs["decoder_logits"].sum() +
        outputs["energies"].sum()
    )
    total_loss.backward()
    
    # Check gradients flow to retrieval_queries
    assert retrieval_queries.grad is not None, "Gradients should flow to retrieval_queries"
    
    # Check gradients flow to trainable model parameters  
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    grad_norm = sum(p.grad.abs().sum().item() for p in trainable_params if p.grad is not None)
    assert grad_norm > 0, "Trainable parameters should have gradients"
