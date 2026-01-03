import os
import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from transformers import DataCollatorWithFlattening

from training.trainer import AccelerateTrainer
from dataset.action_dataset import ActionDataset
from models.action_vae import VariationalAutoEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequencePackedCollator(DataCollatorWithFlattening):
    def __init__(
        self,
        n_latents: int = 64,
        return_position_ids: bool = True,
        separator_id: int = -100,
        return_flash_attn_kwargs: bool = True,
        return_seq_idx: bool = False,
        return_tensors: str = "pt",
    ):
        super().__init__(
            return_position_ids=return_position_ids,
            separator_id=separator_id,
            return_flash_attn_kwargs=return_flash_attn_kwargs,
            return_seq_idx=return_seq_idx,
            return_tensors=return_tensors,
        )
        self.n_latents = n_latents

    def __call__(
        self,
        features: Dict,
        return_tensors: bool = None,
        separator_id: int = None,
    ):

        batch = super().__call__(features, return_tensors, separator_id)

        batch["input_ids"] = batch["input_ids"].squeeze(0)
        batch["position_ids"] = batch["position_ids"].squeeze(0)

        batch["adjusted_position_ids"] = self._adjust_position_ids(
            cu_seq_lens=batch["cu_seq_lens_q"],
        )

        batch["latent_indices"], batch["token_indices"] = (
            self._compute_interleave_indices(
                cu_seq_lens=batch["cu_seq_lens_q"],
            )
        )

        batch["max_seq_len_k"] = batch["max_length_k"]

        batch.pop("labels")
        batch.pop("max_length_q")
        batch.pop("max_length_k")

        return batch

    def _adjust_position_ids(
        self,
        cu_seq_lens: torch.Tensor,
    ) -> torch.Tensor:

        position_ids = []

        for i in range(len(cu_seq_lens) - 1):
            seq_len = cu_seq_lens[i + 1] - cu_seq_lens[i]

            position_ids.append(
                torch.arange(0, self.n_latents + seq_len, dtype=torch.long)
            )

        return torch.cat(position_ids, dim=0)

    def _compute_interleave_indices(self, cu_seq_lens: torch.Tensor):
        """
        Compute indices for vectorized interleaving of latents and tokens.

        Returns:
            latent_indices: [batch_size * n_latents] - where to place latents
            token_indices: [total_tokens] - where to place tokens
        """
        batch_size = len(cu_seq_lens) - 1

        latent_indices = []
        token_indices = []
        current_pos = 0

        for i in range(batch_size):
            # Latents first
            latent_indices.extend(range(current_pos, current_pos + self.n_latents))
            current_pos += self.n_latents

            # Then tokens
            num_tokens = (cu_seq_lens[i + 1] - cu_seq_lens[i]).item()
            token_indices.extend(range(current_pos, current_pos + num_tokens))
            current_pos += num_tokens

        return (
            torch.tensor(latent_indices, dtype=torch.long),
            torch.tensor(token_indices, dtype=torch.long),
        )


class PackedBatchSampler:
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        max_tokens: int = 16384,
    ):
        self.dataset = dataset
        self.max_tokens = max_tokens

    def __iter__(self):
        current_batch = []
        current_tokens = 0

        for idx in range(len(self.dataset)):
            seq_len = len(self.dataset[idx]["input_ids"])

            if current_tokens + seq_len > self.max_tokens and current_batch:
                yield current_batch
                current_batch = []
                current_tokens = 0

            current_batch.append(idx)
            current_tokens += seq_len

        if current_batch:
            yield current_batch


class VAETrainer(AccelerateTrainer):
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        data_collate_fn: Optional[DataCollatorWithFlattening] = None,
        batch_sampler: Optional[Callable] = None,
        eval_fn: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
    ):

        super().__init__(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            collate_fn=data_collate_fn,
            batch_sampler=batch_sampler,
            eval_fn=eval_fn,
            tokenizer=tokenizer,
        )

        self.kl_anneal_steps = config.kl_anneal_steps
        self.free_bits = config.free_bits
        self.data_collate_fn = data_collate_fn

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
        """Compute VAE loss (reconstruction + KL divergence)."""

        input_ids = batch["input_ids"]

        if self.data_collate_fn is not None:
            # Sequence packed
            position_ids = batch["position_ids"]
            adjusted_position_ids = batch["adjusted_position_ids"]
            latent_indices = batch["latent_indices"]
            token_indices = batch["token_indices"]
            max_seq_len_k = batch["max_seq_len_k"]
            cu_seq_lens_k = batch["cu_seq_lens_k"]

            outputs = self.model(
                input_ids=input_ids,
                max_seq_len=max_seq_len_k,
                cu_seq_lens=cu_seq_lens_k,
                token_indices=token_indices,
                latent_indices=latent_indices,
                position_ids=position_ids,
                adjusted_position_ids=adjusted_position_ids,
            )

            logits = outputs["logits"]

            targets = input_ids

            mask = targets != self.data_collate_fn.separator_id

            recon_loss = F.cross_entropy(logits[mask], targets[mask], reduction="mean")

            kl_per_dim = -0.5 * (
                1 + outputs["log_var"] - outputs["mean"] ** 2 - outputs["log_var"].exp()
            )  # [total_latents, d_latent]

            # Sum over latent dimensions
            kl_per_latent = kl_per_dim.sum(dim=-1)  # [total_latents]

            # Reshape from packed to batched [batch_size, n_latents]
            total_latents = kl_per_latent.shape[0]
            n_latents = self.model.n_latents
            batch_size = total_latents // n_latents
            kl_per_latent = kl_per_latent.view(batch_size, n_latents)

            # Apply free bits per latent
            kl_per_latent = torch.max(
                kl_per_latent, torch.tensor(self.free_bits, device=kl_per_latent.device)
            )

            # Sum over latents per sequence, mean over batch
            kl_loss = kl_per_latent.mean()
        else:
            # Batched
            attention_mask = batch["attention_mask"]

            outputs = self.model(input_ids=input_ids, attn_mask=attention_mask)

            logits = outputs["logits"]  # [batch, seq_len, vocab_size]
            targets = input_ids  # [batch, seq_len]

            vocab_size = logits.shape[-1]

            # Flatten for loss computation
            logits_flat = logits.reshape(-1, vocab_size)
            targets_flat = targets.reshape(-1)
            mask = targets_flat != self.tokenizer.pad_token_id

            recon_loss = F.cross_entropy(
                logits_flat[mask], targets_flat[mask], reduction="mean"
            )

            kl_per_dim = -0.5 * (
                1 + outputs["log_var"] - outputs["mean"] ** 2 - outputs["log_var"].exp()
            )  # [batch, n_latents, d_latent]

            # Sum over latent dimensions
            kl_per_latent = kl_per_dim.sum(dim=-1)  # [batch, n_latents]

            # Apply free bits per latent
            kl_per_latent = torch.max(
                kl_per_latent, torch.tensor(self.free_bits, device=kl_per_latent.device)
            )  # [batch, n_latents]

            # Sum over latents per sequence, mean over batch
            kl_loss = kl_per_latent.mean()

        # Anneal KL weight
        kl_weight = self.get_kl_weight()

        # Total loss
        loss = recon_loss + kl_weight * kl_loss

        metrics = {
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "kl_weight": kl_weight,
        }

        # Add logging in your loss computation:
        # mean_norm = outputs["mean"].abs().mean()
        # log_var_mean = outputs["log_var"].mean()
        # variance_mean = outputs["log_var"].exp().mean()

        # logging.info(
        #     f"mean_abs: {mean_norm:.3f}, log_var: {log_var_mean:.3f}, variance: {variance_mean:.3f}"
        # )

        return loss, metrics

    def _default_eval_fn(self, eval_dataloader):
        """Override eval to add generation samples."""

        eval_metrics = super()._default_eval_fn(eval_dataloader)

        if self.accelerator.is_main_process:
            self._log_generation_samples(n_samples=5)

        return eval_metrics

    def _log_generation_samples(self, n_samples: int = 5):
        """Generate and log samples from random latents."""

        logging.info("=" * 60)
        logging.info("GENERATION SAMPLES FROM RANDOM LATENTS")
        logging.info("=" * 60)

        self.model.eval()

        with torch.no_grad():
            for i in range(n_samples):
                # Sample random latent (packed format: [n_latents, d_latent])
                z = torch.randn(
                    self.model.n_latents,
                    self.model.d_latent,
                    device=self.accelerator.device,
                    dtype=torch.bfloat16,
                )

                try:
                    # Generate from latent
                    generated_ids = self.model.generative_model.generate(
                        latent=z,
                        max_length=512,
                        temperature=0.8,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                    # Decode to text
                    text = self.tokenizer.decode(
                        generated_ids[0], skip_special_tokens=True
                    )

                except Exception as e:
                    text = f"[Generation failed: {str(e)}]"
                    import traceback

                    logging.error(traceback.format_exc())

                logging.info(f"\nSample {i}:")
                logging.info(text)
                logging.info("-" * 60)

        logging.info("=" * 60)

        # Return to train mode
        self.model.train()


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

    # dataset = ActionDataset.build_dataset(cfg.dataset)

    # dl = torch.utils.data.DataLoader(
    #     dataset,
    #     collate_fn=SequencePackedCollator(),
    #     batch_size=4,
    # )

    # batch = next(iter(dl))

    # assert False

    dataset = ActionDataset.load(
        src_dir=Path(
            cfg.training.cache_dir,
            cfg.training.dataset_name,
        )
    )

    if cfg.training.show_stats and local_rank == 0:
        logging.info("Calculating dataset statistics...")
        logging.info(dataset.stats)

    eval_size = 20_000  # int(len(dataset) * 0.1)
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

    # datacollator = SequencePackedCollator(
    #     n_latents=cfg.model.action_model.n_action_latents, return_flash_attn_kwargs=True
    # )

    # sampler = PackedBatchSampler(dataset, max_tokens=cfg.training.max_tokens_per_batch)

    datacollator = None
    batch_sampler = None

    trainer = VAETrainer(
        model=model,
        config=cfg.training,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collate_fn=datacollator,
        tokenizer=dataset.tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    main()
