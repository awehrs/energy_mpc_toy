import copy
import os
import time
import logging
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional

import einops
import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from models.agent import ToolAgent
from models.sensors import LanguageSensor
from models.memory import GatedDeltaMemory
from models.predictor import JEPAPredictor
from training.trainer import AccelerateTrainer
from dataset.trajectory_dataset import TrajectoryDataset
from dataset.collation import BucketBatchSampler, TrajectoryCollator


logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="Removed shared tensor.*while saving")


class EMAEncoder(nn.Module):
    """EMA copy of sensor projection + precept downsampler for JEPA targets.

    Shares the frozen encoder backbone with the student to save memory.
    Only maintains EMA copies of trainable components (projection + downsampler).
    """

    def __init__(self, sensor: nn.Module, downsampler: nn.Module, decay: float = 0.999):
        super().__init__()
        self.decay = decay

        # Share frozen encoder (not EMA'd)
        self.encoder = sensor.encoder

        # EMA copies of trainable components
        self.projection = copy.deepcopy(sensor.projection)
        self.downsampler = copy.deepcopy(downsampler)

        self.projection.requires_grad_(False)
        self.downsampler.requires_grad_(False)

    @torch.no_grad()
    def update(self, sensor: nn.Module, downsampler: nn.Module) -> None:
        for ema_p, student_p in zip(
            self.projection.parameters(), sensor.projection.parameters()
        ):
            ema_p.data.mul_(self.decay).add_((1.0 - self.decay) * student_p.data)

        for ema_p, student_p in zip(
            self.downsampler.parameters(), downsampler.parameters()
        ):
            ema_p.data.mul_(self.decay).add_((1.0 - self.decay) * student_p.data)

    @torch.no_grad()
    def forward(
        self,
        precept_tokens: torch.Tensor,
        precept_position_ids: torch.Tensor,
        precept_max_seq_len: int,
        precept_cu_seq_lens: torch.Tensor,
        cu_traj_lens: torch.Tensor,
        bsz: int,
        max_traj_len: int,
    ) -> torch.Tensor:
        """Produce EMA target latents [B, T, Np, d]."""

        input_ids = einops.rearrange(precept_tokens, "t -> 1 t")
        position_ids = einops.rearrange(precept_position_ids, "t -> 1 t")

        hidden = self.encoder.model(
            input_ids=input_ids,
            position_ids=position_ids,
        ).last_hidden_state

        hidden = self.projection(hidden)
        hidden = einops.rearrange(hidden, "1 t d -> t d")

        latents = self.downsampler(
            embedding=hidden,
            max_seq_len=precept_max_seq_len,
            cu_seq_lens=precept_cu_seq_lens,
        )

        # Unpack packed → padded [B, T, Np, d]
        n_latents = self.downsampler.query.shape[0]
        d = latents.shape[-1]
        per_step = latents.view(-1, n_latents, d)
        padded = latents.new_zeros(bsz, max_traj_len, n_latents, d)
        for i in range(bsz):
            start = cu_traj_lens[i]
            end = cu_traj_lens[i + 1]
            padded[i, : end - start] = per_step[start:end]

        return padded


class JepaTrainer(AccelerateTrainer):
    """Trainer for JEPA with EMA target encoder."""

    def __init__(
        self,
        jepa_model: nn.Module,
        config: DictConfig,
        train_dataset: torch.utils.data.Dataset,
        max_traj_len: int,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        collate_fn: Optional[Callable] = None,
        train_batch_sampler: Optional[Callable] = None,
        eval_batch_sampler: Optional[Callable] = None,
        eval_fn: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
    ):

        super().__init__(
            model=jepa_model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            collate_fn=collate_fn,
            train_batch_sampler=train_batch_sampler,
            eval_batch_sampler=eval_batch_sampler,
            eval_fn=eval_fn,
            tokenizer=tokenizer,
        )

        # EMA encoder: shares frozen backbone, EMA-copies projection + downsampler.
        # Created after super().__init__ so params are already on device.
        self.ema_encoder = EMAEncoder(
            sensor=jepa_model.sensor,
            downsampler=jepa_model.precept_downsampler,
            decay=getattr(config, "ema_decay", 0.999),
        )

        # Keep unwrapped reference (unwrap_model chokes on partial compilation)
        self._unwrapped_model = jepa_model

        # Pre-compile flex_attention + torch.compile graph for the predictor
        # Runs through the actual compiled predictor for all expected T values
        logger.info("Warming up predictor (flex_attention + torch.compile)...")
        t0 = time.time()
        n_state = jepa_model.precept_downsampler.query.shape[0]
        n_action = jepa_model.action_downsampler.query.shape[0]
        device = self.accelerator.device
        predictor = self.model.predictor  # compiled version
        self.max_traj_len = max_traj_len
        with torch.no_grad(), self.accelerator.autocast():
            for T in range(2, max_traj_len + 1):
                B = 2
                d = jepa_model.predictor.d_model
                state = torch.zeros(
                    B, T, n_state, d, device=device, dtype=torch.bfloat16
                )
                action = torch.zeros(
                    B, T, n_action, d, device=device, dtype=torch.bfloat16
                )
                traj_mask = torch.ones(B, T, dtype=torch.bool, device=device)
                block_mask = JEPAPredictor.build_block_mask(
                    traj_mask, n_state + n_action, device
                )
                predictor(state=state, block_mask=block_mask, action=action)
                torch.cuda.synchronize()
        logger.info(f"Predictor warmup done in {time.time() - t0:.1f}s")

    def _compile(self):
        """Compile only the predictor — it needs torch.compile for flex_attention.

        Memory (GDN) is excluded: flash-linear-attention already provides optimised
        CUDA kernels, and GDN sees variable (B*T, N, d) shapes per micro-batch which
        causes constant recompilation overhead.
        """
        self.model.predictor = torch.compile(
            self.model.predictor,
            mode=self.config.torch_compile_mode,
        )
        return self.model

    def save_checkpoint(self, step: int, is_best: bool = False):
        super().save_checkpoint(step, is_best)
        if self.accelerator.is_main_process:
            model = self._unwrapped_model
            checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
            model.save_components(checkpoint_dir, self.model_config)
            if is_best:
                best_dir = Path(self.config.output_dir) / "best_model"
                model.save_components(best_dir, self.model_config)

    def _default_eval_fn(self, eval_dataloader) -> Dict[str, float]:
        """Evaluate on the full eval set with input ablation diagnostics."""
        self.model.eval()
        total_loss = 0.0
        total_loss_no_action = 0.0
        total_loss_no_state = 0.0
        metric_sums: Dict[str, float] = {}
        num_batches = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                bsz = len(batch["cu_traj_lens"]) - 1

                # EMA targets (shared across all runs)
                with self.accelerator.autocast():
                    targets = self.ema_encoder(
                        precept_tokens=batch["precept_ids"],
                        precept_position_ids=batch["precept_position_ids"],
                        precept_max_seq_len=batch["precept_max_seq_len"],
                        precept_cu_seq_lens=batch["precept_cu_seq_lens"],
                        cu_traj_lens=batch["cu_traj_lens"],
                        bsz=bsz,
                        max_traj_len=batch["max_traj_len"],
                    )

                fwd_kwargs = dict(
                    precept_tokens=batch["precept_ids"],
                    action_tokens=batch["action_ids"],
                    precept_max_seq_len=batch["precept_max_seq_len"],
                    precept_cu_seq_lens=batch["precept_cu_seq_lens"],
                    precept_position_ids=batch["precept_position_ids"],
                    action_max_seq_len=batch["action_max_seq_len"],
                    action_cu_seq_lens=batch["action_cu_seq_lens"],
                    action_position_ids=batch["action_position_ids"],
                    max_traj_len=batch["max_traj_len"],
                    cu_traj_lens=batch["cu_traj_lens"],
                    traj_mask=batch["traj_mask"],
                    targets=targets,
                )

                # Normal eval
                with self.accelerator.autocast():
                    outputs = self.model.jepa_forward(**fwd_kwargs)
                total_loss += outputs["loss"].item()

                # Ablation: zero out action tokens
                with self.accelerator.autocast():
                    fwd_kwargs["action_tokens"] = torch.zeros_like(batch["action_ids"])
                    out_no_action = self.model.jepa_forward(**fwd_kwargs)
                total_loss_no_action += out_no_action["loss"].item()

                # Ablation: zero out precept tokens (restore action)
                with self.accelerator.autocast():
                    fwd_kwargs["action_tokens"] = batch["action_ids"]
                    fwd_kwargs["precept_tokens"] = torch.zeros_like(
                        batch["precept_ids"]
                    )
                    out_no_state = self.model.jepa_forward(**fwd_kwargs)
                total_loss_no_state += out_no_state["loss"].item()

                for k, v in outputs.items():
                    if k in ("loss", "prediction", "state"):
                        continue
                    val = v.item() if hasattr(v, "item") else v
                    metric_sums[k] = metric_sums.get(k, 0.0) + val
                num_batches += 1

        n = max(num_batches, 1)
        results = {
            "eval_loss": total_loss / n,
            "eval_loss_no_action": total_loss_no_action / n,
            "eval_loss_no_state": total_loss_no_state / n,
        }
        for k, v in metric_sums.items():
            results[k] = v / n

        self.model.train()
        return results

    def on_optimizer_step_end(self):
        super().on_optimizer_step_end()
        logger.info(f"[step {self.global_step}] EMA update start")
        model = self._unwrapped_model
        self.ema_encoder.update(model.sensor, model.precept_downsampler)
        logger.info(f"[step {self.global_step}] EMA update done")

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        bsz = len(batch["cu_traj_lens"]) - 1
        logger.info(
            f"[step {self.global_step}] _compute_loss: bsz={bsz}, "
            f"precept_tokens={batch['precept_ids'].shape}, "
            f"action_tokens={batch['action_ids'].shape}, "
            f"max_traj_len={batch['max_traj_len']}"
        )

        # 1. EMA forward → targets [B, T, Np, d]
        t0 = time.time()
        targets = self.ema_encoder(
            precept_tokens=batch["precept_ids"],
            precept_position_ids=batch["precept_position_ids"],
            precept_max_seq_len=batch["precept_max_seq_len"],
            precept_cu_seq_lens=batch["precept_cu_seq_lens"],
            cu_traj_lens=batch["cu_traj_lens"],
            bsz=bsz,
            max_traj_len=batch["max_traj_len"],
        )
        torch.cuda.synchronize()
        logger.info(f"  EMA forward: {time.time() - t0:.2f}s")

        # Overwrite EMA target at the terminal (STOP) step with s_terminal.
        # Each trajectory's last valid step should predict the constant terminal state.
        s_terminal = self._unwrapped_model.s_terminal  # [Np, d]
        cu_traj_lens = batch["cu_traj_lens"]
        for i in range(bsz):
            last_step = (cu_traj_lens[i + 1] - cu_traj_lens[i] - 1).item()
            targets[i, last_step] = s_terminal.detach()

        # 2. Student forward (loss computed inside jepa_forward)
        t0 = time.time()
        outputs = self.model.jepa_forward(
            precept_tokens=batch["precept_ids"],
            action_tokens=batch["action_ids"],
            precept_max_seq_len=batch["precept_max_seq_len"],
            precept_cu_seq_lens=batch["precept_cu_seq_lens"],
            precept_position_ids=batch["precept_position_ids"],
            action_max_seq_len=batch["action_max_seq_len"],
            action_cu_seq_lens=batch["action_cu_seq_lens"],
            action_position_ids=batch["action_position_ids"],
            max_traj_len=batch["max_traj_len"],
            cu_traj_lens=batch["cu_traj_lens"],
            traj_mask=batch["traj_mask"],
            targets=targets,
        )
        torch.cuda.synchronize()
        logger.info(f"  Student forward: {time.time() - t0:.2f}s")

        # Diagnostics
        with torch.no_grad():
            # Effective rank of target representations
            flat = targets[batch["traj_mask"]].reshape(-1, targets.shape[-1]).float()
            flat = flat - flat.mean(0)
            s = torch.linalg.svdvals(flat)
            p = s / s.sum()
            eff_rank = torch.exp(-(p * p.clamp(min=1e-12).log()).sum())

            # Cosine similarity between consecutive target steps
            traj_mask = batch["traj_mask"]
            consec_mask = traj_mask[:, :-1] & traj_mask[:, 1:]  # [B, T-1]
            t_cur = targets[:, :-1].flatten(2)  # [B, T-1, Np*d]
            t_nxt = targets[:, 1:].flatten(2)  # [B, T-1, Np*d]
            cos = nn.functional.cosine_similarity(t_cur, t_nxt, dim=-1)  # [B, T-1]
            target_consec_cos = (cos * consec_mask).sum() / consec_mask.sum().clamp(
                min=1
            )

        return outputs["loss"], {
            "target_std": targets.std().detach(),
            "target_eff_rank": eff_rank,
            "target_consec_cos": target_consec_cos,
            "jepa_loss": outputs.get("jepa_loss", outputs["loss"].detach()),
            "vic_var": outputs.get("vic_var", torch.tensor(0.0)),
            "vic_cov": outputs.get("vic_cov", torch.tensor(0.0)),
        }


# Find config directory - handle both local and container environments
script_dir = Path(__file__).parent
if (script_dir / "conf").exists():
    # Container environment: config is next to script
    config_path = str(script_dir / "conf")
else:
    # Local environment: config is at project root
    project_root = script_dir.parent
    config_path = str(project_root / "conf")


@hydra.main(version_base="1.1", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    # dataset = TrajectoryDataset.build_dataset(cfg.dataset)

    # eval_size = 1
    # train_size = len(dataset) - eval_size

    # generator = torch.Generator().manual_seed(cfg.training.seed)

    # train_dataset, eval_dataset = torch.utils.data.random_split(
    #     dataset,
    #     lengths=[train_size, eval_size],
    #     generator=generator,
    # )

    # train_batch_sampler = BucketBatchSampler(
    #     train_dataset,
    #     max_tokens=cfg.training.max_tokens_per_batch,
    # )

    # dl = torch.utils.data.DataLoader(
    #     dataset,
    #     collate_fn=TrajectoryCollator(pad_token_id=dataset.tokenizer.pad_token_id),
    #     batch_size=4,
    # )

    # batch = next(iter(dl))

    # assert False

    dataset = TrajectoryDataset.load(
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

    # --- Build model ---

    sensor = LanguageSensor(
        model_name=cfg.model.sensor.model_name,
        d_model=cfg.model.d_model,
    )

    memory = GatedDeltaMemory(
        d_model=cfg.model.d_model,
        n_state_latents=cfg.model.n_precept_latents,
        n_heads=cfg.model.n_heads,
        n_blocks=cfg.model.memory.n_blocks,
        n_mixer_layers=cfg.model.memory.n_mixer_layers,
        norm_eps=cfg.model.norm_eps,
    )

    predictor = JEPAPredictor(
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.predictor.n_layers,
        norm_eps=cfg.model.norm_eps,
    )

    model = ToolAgent(
        config=cfg,
        sensor=sensor,
        memory=memory,
        predictor=predictor,
    )

    # --- Dataloading ---

    collator = TrajectoryCollator(
        pad_token_id=dataset.tokenizer.pad_token_id,
    )

    train_batch_sampler = BucketBatchSampler(
        train_dataset,
        max_tokens=cfg.training.max_tokens_per_batch,
    )

    eval_batch_sampler = BucketBatchSampler(
        eval_dataset,
        max_tokens=cfg.training.max_tokens_per_batch,
        shuffle=False,
        drop_last=False,
    )

    # --- Train ---

    trainer = JepaTrainer(
        jepa_model=model,
        config=cfg.training,
        train_dataset=train_dataset,
        max_traj_len=dataset.stats["trajectory"]["max"],
        eval_dataset=eval_dataset,
        collate_fn=collator,
        train_batch_sampler=train_batch_sampler,
        eval_batch_sampler=eval_batch_sampler,
        tokenizer=dataset.tokenizer,
    )

    # Start training
    logger.info("Starting training loop...")
    trainer.train()

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
