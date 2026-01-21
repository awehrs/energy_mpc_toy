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
        max_action_len: int = 2048,
        return_position_ids: bool = True,
        pad_token_id: int = None,
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
        self.pad_token_id = pad_token_id
        self.max_action_len = max_action_len

    def __call__(
        self,
        features: Dict,
        return_tensors: bool = None,
        separator_id: int = None,
    ):
        batch = super().__call__(features, return_tensors, separator_id)

        data = self._create_decoder_packed_data(cu_seqlens=batch["cu_seq_lens_q"])
        batch.update(data)

        batch["input_ids"] = batch["input_ids"].squeeze(0)
        batch["position_ids"] = batch["position_ids"].squeeze(0)
        batch["max_seq_len_k"] = batch["max_length_k"]

        batch.pop("max_length_q")
        batch.pop("max_length_k")

        batch_size = len(batch["cu_seq_lens_q"]) - 1

        logging.info(f"Num examples: {batch_size}")

        return batch

    def _create_decoder_packed_data(
        self,
        cu_seqlens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Create packed padding and position IDs for decoder"""

        batch_size = len(cu_seqlens) - 1

        all_decoder_position_ids = []
        latent_indices = []
        padding_indices = []
        decoder_cu_seqlens = [0]
        current_global_idx = 0

        for i in range(batch_size):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seq_len = end - start
            pad_len = min(seq_len, self.max_action_len)

            total_seq_len = self.n_latents + pad_len

            pos_ids = torch.arange(0, total_seq_len, dtype=torch.long)
            all_decoder_position_ids.append(pos_ids)

            latent_indices.extend(
                range(current_global_idx, current_global_idx + self.n_latents)
            )
            padding_start = current_global_idx + self.n_latents
            padding_end = current_global_idx + total_seq_len
            padding_indices.extend(range(padding_start, padding_end))

            current_global_idx += total_seq_len
            decoder_cu_seqlens.append(current_global_idx)

        return {
            "latent_indices": torch.tensor(latent_indices, dtype=torch.long),
            "padding_indices": torch.tensor(padding_indices, dtype=torch.long),
            "decoder_position_ids": torch.cat(all_decoder_position_ids),
        }

    def unpack_and_pad_targets(
        self,
        targets: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_length: int,
        pad_token_id: int,
    ) -> torch.Tensor:
        """Unpack targets and pad each sequence to max_length"""

        packed_targets = targets.squeeze(0)
        batch_size = len(cu_seqlens) - 1

        padded_list = []

        for i in range(batch_size):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seq = packed_targets[start:end]

            padding = torch.full(
                (max_length - len(seq),),
                pad_token_id,
                dtype=seq.dtype,
                device=seq.device,
            )
            seq = torch.cat([seq, padding])

            padded_list.append(seq)

        return torch.stack(padded_list)


class PackedBatchSampler:

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        max_tokens: int = 16384,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed

        # Get distributed info
        self.num_replicas = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        self.epoch = 0

        # Handle Subset vs regular dataset
        if isinstance(dataset, torch.utils.data.Subset):
            actual_dataset = dataset.dataset
            self.subset_indices = list(dataset.indices)
            all_lengths = list(actual_dataset.data["token_length"])
            self.seq_lengths = [all_lengths[i] for i in self.subset_indices]
        else:
            self.subset_indices = None
            self.seq_lengths = list(dataset.data["token_length"])

    def __iter__(self):

        # Shuffle indices
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.seq_lengths), generator=g).tolist()
            assert all(
                0 <= idx < len(self.seq_lengths) for idx in indices
            ), f"Shuffle created invalid indices! Max: {max(indices)}, len(seq_lengths): {len(self.seq_lengths)}"
        else:
            indices = list(range(len(self.seq_lengths)))

        # Pack sequences into batches
        all_batches = []
        current_batch = []
        current_tokens = 0

        for idx in indices:
            seq_len = self.seq_lengths[idx]

            if current_tokens + seq_len > self.max_tokens and current_batch:
                all_batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(idx)
            current_tokens += seq_len

        if current_batch:
            all_batches.append(current_batch)

        # Split batches across GPUs
        batches_per_replica = len(all_batches) // self.num_replicas
        start_idx = self.rank * batches_per_replica
        end_idx = (
            start_idx + batches_per_replica
            if self.rank < self.num_replicas - 1
            else len(all_batches)
        )

        # Yield batches, mapping back to original indices if needed
        for batch in all_batches[start_idx:end_idx]:
            yield batch

    def __len__(self):
        # Estimate number of batches
        avg_seq_len = sum(self.seq_lengths) / len(self.seq_lengths)
        estimated_batches = (len(self.seq_lengths) * avg_seq_len) // self.max_tokens
        return int(estimated_batches // self.num_replicas)

    def set_epoch(self, epoch):
        """Call this at the start of each epoch for proper shuffling"""
        self.epoch = epoch


class VAETrainer(AccelerateTrainer):
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        data_collate_fn: Optional[DataCollatorWithFlattening] = None,
        eval_batch_sampler: Optional[Callable] = None,
        train_batch_sampler: Optional[Callable] = None,
        eval_fn: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
    ):

        super().__init__(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            collate_fn=data_collate_fn,
            eval_batch_sampler=eval_batch_sampler,
            train_batch_sampler=train_batch_sampler,
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
            max_seq_len_k = batch["max_seq_len_k"]
            cu_seq_lens_k = batch["cu_seq_lens_k"]
            latent_indices = batch["latent_indices"]
            padding_indices = batch["padding_indices"]
            decoder_position_ids = batch["decoder_position_ids"]

            with self.accelerator.autocast():

                outputs = self.model(
                    input_ids=input_ids,
                    max_seq_len=max_seq_len_k,
                    cu_seq_lens=cu_seq_lens_k,
                    position_ids=position_ids,
                    latent_indices=latent_indices,
                    padding_indices=padding_indices,
                    decoder_position_ids=decoder_position_ids,
                    targets=batch["labels"],
                )

                recon_loss = outputs["loss"]

        else:
            # Batched
            attention_mask = batch["attention_mask"]

            outputs = self.model(input_ids=input_ids, attn_mask=attention_mask)

            raise NotImplementedError

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
    #     collate_fn=SequencePackedCollator(pad_token_id=dataset.tokenizer.pad_token_id),
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
        vocab_size=len(dataset.tokenizer),
        pad_token_id=dataset.tokenizer.pad_token_id,
        max_action_len=model_cfg.max_action_len,
        n_self_attn_heads=model_cfg.n_self_attn_heads,
        n_cross_attn_heads=model_cfg.n_cross_attn_heads,
        n_encoder_self_attn_layers=model_cfg.n_encoder_self_attn_layers,
        n_decoder_self_attn_layers=model_cfg.n_decoder_self_attn_layers,
        dropout=model_cfg.dropout,
    ).to(torch.bfloat16)

    datacollator = SequencePackedCollator(
        n_latents=cfg.model.action_model.n_action_latents,
        max_action_len=cfg.max_action_len,
        pad_token_id=dataset.tokenizer.pad_token_id,
        return_flash_attn_kwargs=True,
    )

    eval_batch_sampler = PackedBatchSampler(
        eval_dataset, max_tokens=cfg.training.max_tokens_per_batch
    )

    train_batch_sampler = PackedBatchSampler(
        train_dataset, max_tokens=cfg.training.max_tokens_per_batch
    )

    trainer = VAETrainer(
        model=model,
        config=cfg.training,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collate_fn=datacollator,
        eval_batch_sampler=eval_batch_sampler,
        train_batch_sampler=train_batch_sampler,
        tokenizer=dataset.tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    main()
