import os
import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from training.trainer import AccelerateTrainer
from dataset.action_dataset import ActionDataset
from models.action_vae import VariationalAutoEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VAETrainer(AccelerateTrainer):
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        eval_fn: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
    ):

        super().__init__(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_fn=eval_fn,
            tokenizer=tokenizer,
        )

        self.kl_anneal_steps = config.kl_anneal_steps
        self.free_bits = config.free_bits

    def _compile(self):

        self.model.recognition_model = torch.compile(
            self.model.recognition_model,
            mode=self.config.torch_compile_mode,
        )
        self.model.generative_model.self_attention_layers = torch.compile(
            self.model.generative_model.self_attention_layers,
            mode=self.config.torch_compile_mode,
        )
        self.model.generative_model.projection = torch.compile(
            self.model.generative_model.projection,
            mode=self.config.torch_compile_mode,
        )

        return self.model

    def get_kl_weight(self) -> float:
        """Linear KL annealing from 0 to 1"""
        return min(1.0, self.global_step / self.kl_anneal_steps)

    def _compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute VAE loss (reconstruction + KL divergence)
        Expected batch keys:
        - input_ids: [batch, seq_len] for encoder
        - attention_mask: [batch, seq_len]
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Forward pass
        outputs = self.model(input_ids, attention_mask)

        # Reconstruction loss (cross entropy)
        logits = outputs["logits"][:, :-1, :]  # [batch, seq-1, vocab]
        targets = input_ids[:, 1:]  # [batch, seq-1]

        batch_size, seq_len, vocab_size = logits.shape

        # Compute loss in chunks to avoid OOM
        chunk_size = 512
        total_loss = 0
        total_tokens = 0

        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)
            logits_chunk = logits[:, i:end_idx, :].reshape(-1, vocab_size)
            targets_chunk = targets[:, i:end_idx].reshape(-1)

            # Mask out padding
            mask = targets_chunk != self.tokenizer.pad_token_id
            if mask.sum() > 0:
                chunk_loss = F.cross_entropy(
                    logits_chunk[mask], targets_chunk[mask], reduction="sum"
                )
                total_loss += chunk_loss
                total_tokens += mask.sum()

        recon_loss = total_loss / total_tokens

        # KL divergence with free bits
        # KL per latent dimension: -0.5 * (1 + log_var - mu^2 - var)
        kl_per_dim = -0.5 * (
            1 + outputs["log_var"] - outputs["mean"] ** 2 - outputs["log_var"].exp()
        )

        # Apply free bits: max(KL, free_bits)
        kl_per_dim = torch.max(
            kl_per_dim, torch.tensor(self.free_bits).to(kl_per_dim.device)
        )

        # Sum over latent dims, mean over batch
        kl_loss = kl_per_dim.sum(dim=-1).mean()

        # Anneal KL weight
        kl_weight = self.get_kl_weight()

        # Total loss
        loss = recon_loss + kl_weight * kl_loss

        logging.info(f"mu dtype: {outputs['mean'].dtype}")
        logging.info(
            f"encoder param dtype: {self.model.recognition_model.input_embedding.weight.dtype}"
        )

        metrics = {
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "kl_weight": kl_weight,
        }

        return loss, metrics


# Find config directory - handle both local and container environments
script_dir = Path(__file__).parent
if (script_dir / "conf").exists():
    # Container environment: config is next to script
    config_path = str(script_dir / "conf")
else:
    # Local environment: config is at project root
    project_root = script_dir.parent
    config_path = str(project_root / "conf")


@hydra.main(version_base=None, config_path=config_path, config_name="config")
def main(cfg: DictConfig):

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    dataset = ActionDataset.load(
        src_dir=Path(
            cfg.training.cache_dir,
            cfg.training.dataset_name,
        )
    )

    if cfg.training.show_stats and local_rank == 0:
        logging.info("Calculating dataset statistics...")
        logging.info(dataset.get_stats())

    eval_size = min(2000, int(len(dataset) * 0.01))
    train_size = len(dataset) - eval_size

    generator = torch.Generator().manual_seed(cfg.training.seed)

    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset,
        lengths=[train_size, eval_size],
        generator=generator,
    )

    model_cfg = cfg.model.action_model

    model = VariationalAutoEncoder(
        config=model_cfg,
        model_name=model_cfg.model_name,
        d_latent=model_cfg.d_latent,
        n_latents=model_cfg.n_action_latents,
        n_self_attn_heads=model_cfg.n_self_attn_heads,
        n_cross_attn_heads=model_cfg.n_cross_attn_heads,
        n_encoder_self_attn_layers=model_cfg.n_encoder_self_attn_layers,
        n_decoder_self_attn_layers=model_cfg.n_decoder_self_attn_layers,
        dropout=model_cfg.dropout,
    ).to(torch.bfloat16)

    trainer = VAETrainer(
        model=model,
        config=cfg.training,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_fn=None,
        tokenizer=dataset.tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    main()
