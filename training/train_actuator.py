import os
import random
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from transformers import DataCollatorWithFlattening
from pydantic.warnings import UnsupportedFieldAttributeWarning

from training.trainer import AccelerateTrainer
from dataset.action_dataset import ActionDataset
from models.sensors import LanguageSensor
from models.actuators import LanguageActuator
from models.attention import Downsampler


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    category=UnsupportedFieldAttributeWarning,
)


class CorruptionScheduler:
    """Trainer sets step; collator queries current corruption knobs."""

    def set_step(self, step: int, total_steps: Optional[int] = None) -> None:
        self.step = step
        self.total_steps = total_steps

    def get(self) -> dict:
        """Return {"corruption_rate": float, "mean_span_length": int}."""
        raise NotImplementedError


@dataclass
class LinearWarmupHold(CorruptionScheduler):
    start_corruption_rate: float = 0.05
    end_corruption_rate: float = 0.5
    warmup_steps: int = 1000
    mean_span_length: int = 3
    full_mask_prob: float = 0.15

    def __post_init__(self):
        self.step = 0
        self.total_steps = None

    def get(self) -> dict:
        s = max(0, int(self.step))
        w = max(1, int(self.warmup_steps))
        frac = min(1.0, s / w)
        rate = self.start_corruption_rate + frac * (
            self.end_corruption_rate - self.start_corruption_rate
        )

        # After warmup, occasionally mask entire sequence
        if frac >= 1.0 and random.random() < self.full_mask_prob:
            rate = 1.0

        return {
            "corruption_rate": float(rate),
            "mean_span_length": int(self.mean_span_length),
        }


class SequencePackedCollator(DataCollatorWithFlattening):
    def __init__(
        self,
        corruption_scheduler: CorruptionScheduler,
        n_latents: int = 64,
        seq_dropout: float = 0.1,
        max_action_len: int = 2048,
        return_position_ids: bool = True,
        mask_token_id: int = None,
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
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.max_action_len = max_action_len
        self.corruption_scheduler = corruption_scheduler

    def __call__(
        self,
        features: Dict,
        return_tensors: bool = None,
        separator_id: int = None,
    ):
        corruption_params = self.corruption_scheduler.get()
        corruption_rate = corruption_params["corruption_rate"]
        mean_span_length = corruption_params["mean_span_length"]

        all_corrupted_ids = []
        all_labels = []

        for sample in features:
            ids = sample["input_ids"]
            if not torch.is_tensor(ids):
                ids = torch.tensor(ids, dtype=torch.long)

            corrupted_ids, labels = self._corrupt_span(
                ids,
                mask_token_id=self.mask_token_id,
                pad_token_id=self.pad_token_id,
                corruption_rate=corruption_rate,
                mean_span_length=mean_span_length,
            )
            all_corrupted_ids.extend(corrupted_ids.tolist())
            all_labels.extend(labels.tolist())

        batch = super().__call__(features, return_tensors, separator_id)

        batch["position_ids"] = self._get_position_ids(
            cu_seqlens=batch["cu_seq_lens_q"]
        ).squeeze(0)
        batch["input_ids"] = batch["input_ids"].squeeze(0)
        batch["max_seq_len"] = batch["max_length_q"]
        batch["cu_seq_lens"] = batch["cu_seq_lens_q"]
        batch["labels"] = torch.tensor(all_labels)
        batch["corrupted_ids"] = torch.tensor(all_corrupted_ids)

        batch.pop("max_length_q")
        batch.pop("max_length_k")
        batch.pop("cu_seq_lens_q")
        batch.pop("cu_seq_lens_k")

        return batch

    def _get_position_ids(
        self,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:

        batch_size = len(cu_seqlens) - 1

        position_ids = []

        for i in range(batch_size):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seq_len = end - start

            pos_ids = torch.arange(0, seq_len, dtype=torch.long)
            position_ids.append(pos_ids)

        return torch.cat(position_ids)

    def _corrupt_span(
        self,
        input_ids: torch.Tensor,
        mask_token_id: int,
        pad_token_id: Optional[int] = None,
        corruption_rate: Union[float, Callable[[int], float]] = 0.15,
        mean_span_length: Union[int, Callable[[int], int]] = 3,
        global_step: int = 0,
        keep_min_unmasked: int = 0,
        special_token_ids: Optional[set] = None,
        rng: Optional[random.Random] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
        corrupted_input_ids: same shape as input_ids
        labels: same shape as input_ids; original tokens where masked, else -100
        """
        assert input_ids.dim() == 1
        if rng is None:
            rng = random

        T = int(input_ids.numel())
        if T == 0:
            return input_ids.clone(), torch.full_like(input_ids, -100)

        # resolve schedules
        rate = float(corruption_rate)
        rate = max(0.0, min(1.0, rate))

        msl = mean_span_length
        msl = max(1, int(msl))

        eligible = torch.ones(T, dtype=torch.bool, device=input_ids.device)

        if pad_token_id is not None:
            eligible &= input_ids != pad_token_id

        if special_token_ids:
            for sid in special_token_ids:
                eligible &= input_ids != sid

        # don't mask position 0 (BOS)
        eligible[0] = False

        eligible_idx = torch.nonzero(eligible, as_tuple=False).view(-1)
        n_eligible = int(eligible_idx.numel())
        if n_eligible == 0 or rate == 0.0:
            return input_ids.clone(), torch.full_like(input_ids, -100)

        max_maskable = max(0, n_eligible - int(keep_min_unmasked))
        target_mask = int(round(rate * n_eligible))
        target_mask = max(0, min(max_maskable, target_mask))
        if target_mask == 0:
            return input_ids.clone(), torch.full_like(input_ids, -100)

        masked = torch.zeros(T, dtype=torch.bool, device=input_ids.device)
        remaining = target_mask
        max_attempts = 10 * (target_mask + 1)
        attempts = 0

        # spans mask contiguous *positions*, but only count eligible tokens
        while remaining > 0 and attempts < max_attempts:
            attempts += 1
            start = int(eligible_idx[rng.randrange(n_eligible)].item())
            span_len = max(1, int(rng.expovariate(1.0 / msl)))

            i = start
            masked_this_span = 0
            while i < T and masked_this_span < span_len and remaining > 0:
                if eligible[i] and not masked[i]:
                    masked[i] = True
                    masked_this_span += 1
                    remaining -= 1
                i += 1

        corrupted = input_ids.clone()
        labels = torch.full_like(input_ids, -100)
        labels[masked] = input_ids[masked]
        corrupted[masked] = mask_token_id

        labels[0] = -100
        return corrupted, labels


class PackedBatchSampler:

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        max_tokens: int = 4096,
        shuffle: bool = True,
        seed: int = 0,
        num_epochs: int = 1,
    ):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
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

        # Pre-compute batch counts for all epochs
        self.batch_counts_per_epoch = {}

        for epoch in range(num_epochs):
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + epoch)
                indices = torch.randperm(len(self.seq_lengths), generator=g).tolist()
                shuffled_lengths = [self.seq_lengths[i] for i in indices]
            else:
                shuffled_lengths = self.seq_lengths

            self.batch_counts_per_epoch[epoch] = self._count_batches(shuffled_lengths)

    def _count_batches(self, lengths):
        num_batches = 0
        current_tokens = 0

        for seq_len in lengths:
            if current_tokens + seq_len > self.max_tokens and current_tokens > 0:
                num_batches += 1
                current_tokens = 0
            current_tokens += seq_len

        if current_tokens > 0:
            num_batches += 1

        return num_batches // self.num_replicas

    def __len__(self):
        return self.batch_counts_per_epoch.get(
            self.epoch, sum(self.seq_lengths) // self.max_tokens // self.num_replicas
        )

    def get_total_steps(self):
        return sum(self.batch_counts_per_epoch.values())

    def set_epoch(self, epoch):
        self.epoch = epoch

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


class FrozenJEPAEncoder(nn.Module):
    """Frozen JEPA sensor + action_downsampler for producing action latents."""

    def __init__(self, sensor: LanguageSensor, action_downsampler: Downsampler):
        super().__init__()
        self.sensor = sensor
        self.action_downsampler = action_downsampler
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        max_seq_len: int,
        cu_seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Encode tokens → latents.

        Args:
            input_ids:    [total_tokens]   — packed flat token ids
            position_ids: [total_tokens]   — per-token position ids
            max_seq_len:  int              — longest sequence in the batch
            cu_seq_lens:  [num_seqs + 1]   — cumulative token counts

        Returns:
            [total_steps * n_action_latents, d]  — packed action latents
        """
        hidden = self.sensor(input_ids=input_ids, position_ids=position_ids)
        latents = self.action_downsampler(
            embedding=hidden,
            max_seq_len=max_seq_len,
            cu_seq_lens=cu_seq_lens,
        )
        return latents


class ActuatorWithEncoder(nn.Module):
    """Wraps frozen JEPA encoder + trainable LanguageActuator for training.

    The config attribute is needed by AccelerateTrainer (for model_config).
    """

    def __init__(
        self,
        config: DictConfig,
        encoder: FrozenJEPAEncoder,
        actuator: LanguageActuator,
    ):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.actuator = actuator

    def forward(
        self,
        input_ids: torch.Tensor,
        corrupted_ids: torch.Tensor,
        position_ids: torch.Tensor,
        max_seq_len: int,
        cu_seq_lens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        latents = self.encoder(
            input_ids=input_ids,
            position_ids=position_ids,
            max_seq_len=max_seq_len,
            cu_seq_lens=cu_seq_lens,
        )
        return self.actuator(
            latents=latents,
            input_ids=corrupted_ids,
            position_ids=position_ids,
            max_seq_len_q=max_seq_len,
            cu_seq_lens_q=cu_seq_lens,
            targets=targets,
        )


class ActuatorTrainer(AccelerateTrainer):
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        corruption_scheduler: CorruptionScheduler,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        collate_fn: Optional[Callable] = None,
        train_batch_sampler: Optional[Callable] = None,
        eval_batch_sampler: Optional[Callable] = None,
        eval_fn: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
    ):
        super().__init__(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            collate_fn=collate_fn,
            train_batch_sampler=train_batch_sampler,
            eval_batch_sampler=eval_batch_sampler,
            eval_fn=eval_fn,
            tokenizer=tokenizer,
        )
        self.corruption_scheduler = corruption_scheduler

    def _compile(self):
        self.model.actuator.self_attention_layers = torch.compile(
            self.model.actuator.self_attention_layers,
            mode=self.config.torch_compile_mode,
        )
        self.model.actuator.projection = torch.compile(
            self.model.actuator.projection,
            mode=self.config.torch_compile_mode,
        )
        return self.model

    def _create_optimizer(self) -> torch.optim.Optimizer:
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            trainable,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay,
        )

    def on_optimizer_step_end(self):
        self.optimizer.step()
        self.lr_scheduler.step()
        self.corruption_scheduler.set_step(self.global_step)
        self.optimizer.zero_grad()

    def train_micro_step(self, batch, timer):
        loss, additional_metrics, grad_norm = super().train_micro_step(batch, timer)

        # Cross-attention gradient norm diagnostic
        decoder = self.model.actuator.decoder
        if hasattr(self.model, "module"):
            decoder = self.model.module.actuator.decoder

        xattn_grad_norm = 0.0
        xattn_grad_count = 0
        for name, p in decoder.named_parameters():
            if "latent_xattn" in name and p.grad is not None:
                xattn_grad_norm += p.grad.data.norm(2).item() ** 2
                xattn_grad_count += 1
        if xattn_grad_count > 0:
            xattn_grad_norm = xattn_grad_norm**0.5

        additional_metrics["xattn_grad_norm"] = xattn_grad_norm

        return loss, additional_metrics, grad_norm

    def save_checkpoint(self, step: int, is_best: bool = False):
        super().save_checkpoint(step, is_best)
        if self.accelerator.is_main_process:
            model = self.accelerator.unwrap_model(self.model)
            checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                model.actuator.state_dict(),
                checkpoint_dir / "actuator.pt",
            )
            if is_best:
                best_dir = Path(self.config.output_dir) / "best_model"
                best_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    model.actuator.state_dict(),
                    best_dir / "actuator.pt",
                )

    def _compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        targets = batch["labels"]
        input_ids = batch["input_ids"]
        corrupted_ids = batch["corrupted_ids"]
        max_seq_len = batch["max_seq_len"]
        cu_seq_lens = batch["cu_seq_lens"]
        position_ids = batch["position_ids"]

        outputs = self.model(
            input_ids=input_ids,
            corrupted_ids=corrupted_ids,
            position_ids=position_ids,
            max_seq_len=max_seq_len,
            cu_seq_lens=cu_seq_lens,
            targets=targets,
        )

        loss = outputs["loss"]

        metrics = {
            "recon_loss": loss.detach(),
        }

        return loss, metrics


@hydra.main(version_base="1.1", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    assert (
        cfg.training.jepa_checkpoint is not None
    ), "Must specify training.jepa_checkpoint for actuator training"
    jepa_dir = Path(cfg.training.jepa_checkpoint)

    # --- Load dataset ---

    dataset = ActionDataset.load(
        src_dir=Path(
            cfg.training.cache_dir,
            cfg.training.dataset_name,
        )
    )

    if cfg.training.show_stats and local_rank == 0:
        logging.info("Calculating dataset statistics...")
        logging.info(dataset.stats)

    eval_size = 20_000
    train_size = len(dataset) - eval_size

    generator = torch.Generator().manual_seed(cfg.training.seed)

    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset,
        lengths=[train_size, eval_size],
        generator=generator,
    )

    # --- Build frozen JEPA encoder ---

    sensor = LanguageSensor(
        model_name=cfg.model.sensor.model_name,
        d_model=cfg.model.d_model,
    )
    sensor.projection.load_state_dict(
        torch.load(jepa_dir / "sensor_projection.pt", weights_only=True)
    )

    action_downsampler = Downsampler(
        d_model=cfg.model.d_model,
        n_latents=cfg.model.n_action_latents,
        n_heads=cfg.model.n_heads,
        norm_eps=cfg.model.norm_eps,
    )
    action_downsampler.load_state_dict(
        torch.load(jepa_dir / "action_downsampler.pt", weights_only=True)
    )

    frozen_encoder = FrozenJEPAEncoder(sensor, action_downsampler)

    # --- Build actuator ---

    actuator = LanguageActuator(
        model_name=cfg.model.sensor.model_name,
        n_latents=cfg.model.n_action_latents,
        latent_dim=cfg.model.d_model,
        pad_token_id=dataset.tokenizer.pad_token_id,
        max_action_len=cfg.max_action_len,
        n_self_attn_layers=cfg.model.action_model.n_decoder_self_attn_layers,
        n_self_attn_heads=cfg.n_self_attn_heads,
        n_cross_attn_heads=cfg.n_cross_attn_heads,
        dropout=cfg.dropout,
    )

    model = ActuatorWithEncoder(
        config=cfg,
        encoder=frozen_encoder,
        actuator=actuator,
    )

    # --- Collation ---

    corruption_scheduler = LinearWarmupHold(
        start_corruption_rate=cfg.training.corruption_start_rate,
        end_corruption_rate=cfg.training.corruption_end_rate,
        warmup_steps=cfg.training.corruption_warmup_steps,
        mean_span_length=cfg.training.corruption_mean_span_length,
        full_mask_prob=cfg.training.corruption_full_mask_prob,
    )

    collator = SequencePackedCollator(
        corruption_scheduler=corruption_scheduler,
        n_latents=cfg.model.n_action_latents,
        max_action_len=cfg.max_action_len,
        mask_token_id=dataset.tokenizer.convert_tokens_to_ids("<|fim_pad|>"),
        pad_token_id=dataset.tokenizer.pad_token_id,
        return_flash_attn_kwargs=True,
    )

    train_batch_sampler = PackedBatchSampler(
        train_dataset,
        max_tokens=cfg.training.max_tokens_per_batch,
    )

    eval_batch_sampler = PackedBatchSampler(
        eval_dataset,
        max_tokens=cfg.training.max_tokens_per_batch,
    )

    # --- Train ---

    trainer = ActuatorTrainer(
        model=model,
        config=cfg.training,
        corruption_scheduler=corruption_scheduler,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        collate_fn=collator,
        train_batch_sampler=train_batch_sampler,
        eval_batch_sampler=eval_batch_sampler,
        tokenizer=dataset.tokenizer,
    )

    logger.info("Starting actuator training...")
    trainer.train()
    logger.info("Actuator training completed!")


if __name__ == "__main__":
    main()
