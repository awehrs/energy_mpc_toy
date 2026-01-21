import os
import time
import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import torch._dynamo
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig
from accelerate import Accelerator
from accelerate.utils import set_seed

torch.set_float32_matmul_precision("high")

torch._dynamo.config.optimize_ddp = False

warnings.filterwarnings("ignore", message=".*Mismatch dtype between input and weight.*")

os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:512"

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    import GPUtil

    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    GPUtil = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Timer:
    def __init__(self, alpha=0.1):
        """
        alpha: smoothing factor
        - Higher alpha (e.g., 0.3) = more weight on recent values
        - Lower alpha (e.g., 0.05) = smoother, slower to adapt
        """
        self.alpha = alpha
        self.fwd_avg = None
        self.bwd_avg = None

    def update_fwd(self, time):
        if self.fwd_avg is None:
            self.fwd_avg = time
        else:
            self.fwd_avg = self.alpha * time + (1 - self.alpha) * self.fwd_avg

    def update_bwd(self, time):
        if self.bwd_avg is None:
            self.bwd_avg = time
        else:
            self.bwd_avg = self.alpha * time + (1 - self.alpha) * self.bwd_avg


class AccelerateTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        collate_fn: Optional[Callable] = None,
        eval_sampler: Optional[Callable] = None,
        train_sampler: Optional[Callable] = None,
        eval_batch_sampler: Optional[Callable] = None,
        train_batch_sampler: Optional[Callable] = None,
        eval_fn: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.config = config
        self.model_config = model.config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.eval_fn = eval_fn or self._default_eval_fn

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                raise ValueError("Tokenizer must have pad token")

        self.tokenizer = tokenizer

        kwargs = {
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "mixed_precision": (
                config.mixed_precision if config.mixed_precision != "none" else None
            ),
            "log_with": "wandb" if config.log_wandb and WANDB_AVAILABLE else None,
            "project_dir": config.output_dir,
        }

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self._param_counts = {"total": total_params, "trainable": trainable_params}
        self.accelerator = Accelerator(**kwargs)

        print(f"=" * 80)
        print(f"Rank {self.accelerator.process_index}:")
        print(f"  LOCAL_RANK env var: {os.environ.get('LOCAL_RANK', 'NOT SET')}")
        print(
            f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}"
        )
        print(f"  torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"  torch.cuda.current_device(): {torch.cuda.current_device()}")
        print(f"  accelerator.device: {self.accelerator.device}")
        print(
            f"  accelerator.local_process_index: {self.accelerator.local_process_index}"
        )
        print(f"  accelerator.num_processes: {self.accelerator.num_processes}")
        print(f"=" * 80)

        self.accelerator.print(f"Accelerator state: {self.accelerator.state}")
        self.accelerator.print(f"Mixed precision: {self.accelerator.mixed_precision}")
        self.accelerator.print(
            f"Gradient accumulation steps: {self.accelerator.gradient_accumulation_steps}"
        )

        # Set random seeds
        set_seed(config.seed)

        self.model = model

        if config.torch_compile:
            self.model = self._compile()

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Create dataloaders
        train_dataloader = self._create_dataloader(
            self.train_dataset,
            is_train=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
            batch_sampler=train_batch_sampler,
        )

        eval_dataloader = None
        if self.eval_dataset is not None:
            eval_dataloader = self._create_dataloader(
                self.eval_dataset,
                is_train=False,
                collate_fn=collate_fn,
                sampler=eval_sampler,
                batch_sampler=eval_batch_sampler,
            )

        # Calculate total training steps
        if config.max_steps > 0:
            self.total_steps = config.max_steps
        else:
            if train_batch_sampler is not None:
                num_batches = len(train_batch_sampler)
                steps_per_epoch = num_batches // config.gradient_accumulation_steps
                self.total_steps = steps_per_epoch * config.num_epochs
            else:
                total_batches = len(self.train_dataset) // (
                    config.per_device_train_batch_size * self.accelerator.num_processes
                )
                steps_per_epoch = total_batches // config.gradient_accumulation_steps
                self.total_steps = steps_per_epoch * config.num_epochs

            # Initialize learning rate scheduler
            self.lr_scheduler = self._create_lr_scheduler(self.total_steps)

        (
            self.model,
            self.optimizer,
            train_dataloader,
            eval_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            train_dataloader,
            eval_dataloader,
            self.lr_scheduler,
        )

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Initialize wandb
        if config.log_wandb and WANDB_AVAILABLE and self.accelerator.is_main_process:
            from omegaconf import OmegaConf

            self.accelerator.init_trackers(
                project_name=config.wandb_project,
                config=OmegaConf.to_object(config),
                init_kwargs=(
                    {
                        "wandb": {
                            "entity": config.wandb_entity,
                            "name": config.wandb_run_name,
                        }
                    }
                    if hasattr(config, "wandb_entity") and config.wandb_entity
                    else {}
                ),
            )

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")

        # Create output directory
        if self.accelerator.is_main_process:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def _compile(self) -> None:
        self.model = torch.compile(
            self.model,
            mode=self.config.torch_compile_mode,
        )

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create the optimizer."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay,
        )

    def _create_lr_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler."""
        if self.config.lr_scheduler_type == "linear":
            from transformers import get_linear_schedule_with_warmup

            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif self.config.lr_scheduler_type == "cosine":
            from transformers import get_cosine_schedule_with_warmup

            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:  # constant
            from transformers import get_constant_schedule_with_warmup

            return get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
            )

    def _default_eval_fn(self, eval_dataloader: DataLoader):
        """Default evaluation function."""
        self.model.eval()
        total_loss = 0
        num_samples = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                with self.accelerator.autocast():
                    loss, additional_metrics = self._compute_loss(batch)

                # Simple accumulation without gathering
                total_loss += loss.item()
                num_samples += 1

        # Average across processes manually
        if self.accelerator.num_processes > 1:
            total_loss = torch.tensor(total_loss, device=self.accelerator.device)
            num_samples = torch.tensor(num_samples, device=self.accelerator.device)

            # Use reduce instead of gather
            torch.distributed.all_reduce(total_loss)
            torch.distributed.all_reduce(num_samples)

            total_loss = total_loss.item() / self.accelerator.num_processes
            num_samples = num_samples.item() / self.accelerator.num_processes

        return {
            "eval_loss": total_loss / num_samples,
            **additional_metrics,
        }

    def _create_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        is_train: bool = True,
        collate_fn: Callable = None,
        sampler: Callable = None,
        batch_sampler: Callable = None,
    ) -> DataLoader:
        """Create dataloader. Can be overridden for custom collation."""

        assert not (
            sampler and batch_sampler
        ), "Cannot use both sampler and batch_sampler"

        if batch_sampler:
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=(
                    self.config.dataloader_pin_memory
                    if torch.cuda.is_available()
                    else False
                ),
            )
        elif sampler:
            return DataLoader(
                dataset,
                batch_size=self._get_batch_size(is_train),
                sampler=sampler,
                collate_fn=collate_fn,
                num_workers=self.config.dataloader_num_workers,
                pin_memory=(
                    self.config.dataloader_pin_memory
                    if torch.cuda.is_available()
                    else False
                ),
            )
        else:
            # Standard dataloader
            return DataLoader(
                dataset,
                batch_size=self._get_batch_size(is_train),
                shuffle=is_train,
                collate_fn=collate_fn,
                num_workers=self.config.dataloader_num_workers,
                pin_memory=(
                    self.config.dataloader_pin_memory
                    if torch.cuda.is_available()
                    else False
                ),
                drop_last=is_train,
            )

    def _get_batch_size(self, is_train=True) -> int:
        """Get batch size for train/eval."""
        return (
            self.config.per_device_train_batch_size
            if is_train
            else self.config.per_device_eval_batch_size
        )

    def _log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log metrics to console and wandb."""
        # Console logging via accelerate (automatically handles main process)
        log_str = f"Step {step}: " + ", ".join(
            [f"{k}: {v:.6f}" for k, v in metrics.items()]
        )
        self.accelerator.print(log_str)

        # Add GPU metrics if available
        if torch.cuda.is_available():
            if GPUTIL_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # First GPU
                        metrics.update(
                            {
                                "gpu_utilization_percent": gpu.load * 100,
                                "gpu_memory_used_gb": gpu.memoryUsed / 1024,
                                "gpu_temperature_c": gpu.temperature,
                            }
                        )
                except:
                    pass

        # Wandb logging via accelerate tracker
        if self.config.log_wandb and WANDB_AVAILABLE:
            self.accelerator.log(metrics, step=step)

    def save_checkpoint(self, step: int, is_best: bool = False):
        """Save model checkpoint."""
        if not self.accelerator.is_main_process:
            return

        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
        self.accelerator.save_state(checkpoint_dir)

        # Save additional training state
        additional_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
        }
        torch.save(additional_state, checkpoint_dir / "training_state.pt")

        if is_best:
            best_dir = Path(self.config.output_dir) / "best_model"
            self.accelerator.save_state(best_dir)
            torch.save(additional_state, best_dir / "training_state.pt")

        # Clean up old checkpoints
        self._cleanup_checkpoints()

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint_dir = Path(checkpoint_path)

        # Load Accelerate state
        self.accelerator.load_state(checkpoint_dir)

        # Load additional training state if available
        training_state_file = checkpoint_dir / "training_state.pt"
        if training_state_file.exists():
            training_state = torch.load(training_state_file, map_location="cpu")
            self.global_step = training_state.get("global_step", 0)
            self.epoch = training_state.get("epoch", 0)
            self.best_eval_loss = training_state.get("best_eval_loss", float("inf"))

        logger.info(
            f"Loaded checkpoint from {checkpoint_path} (step {self.global_step})"
        )

    def _cleanup_checkpoints(self):
        """Remove old checkpoints to respect save_total_limit."""
        if self.config.save_total_limit <= 0:
            return

        checkpoint_dir = Path(self.config.output_dir)
        checkpoints = list(checkpoint_dir.glob("checkpoint-*"))

        if len(checkpoints) > self.config.save_total_limit:
            # Sort by step number
            checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
            # Remove oldest checkpoints
            for checkpoint in checkpoints[: -self.config.save_total_limit]:
                import shutil

                shutil.rmtree(checkpoint)

    def train(self):
        """Main training loop."""
        # Load checkpoint if specified
        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)

        # Log training setup info
        if self.accelerator.is_main_process:
            total_params = self._param_counts["total"]
            trainable_params = self._param_counts["trainable"]

            # Count training examples
            num_train_examples = len(self.train_dataset)

            # Log the info
            logger.info(f"Training setup:")
            logger.info(f"  Total parameters: {total_params:,}")
            logger.info(f"  Trainable parameters: {trainable_params:,}")
            logger.info(f"  Training examples: {num_train_examples:,}")
            logger.info(f"  Total training steps: {self.total_steps:,}")
            logger.info(
                f"  Effective batch size: {self.config.per_device_train_batch_size * self.accelerator.num_processes * self.config.gradient_accumulation_steps}"
            )

            # Log to wandb if available
            if self.config.log_wandb and WANDB_AVAILABLE:
                self.accelerator.log(
                    {
                        "setup/total_parameters": total_params,
                        "setup/trainable_parameters": trainable_params,
                        "setup/num_train_examples": num_train_examples,
                        "setup/total_training_steps": self.total_steps,
                        "setup/effective_batch_size": self.config.per_device_train_batch_size
                        * self.accelerator.num_processes
                        * self.config.gradient_accumulation_steps,
                    },
                    step=0,
                )

        # Training loop
        self.model.train()
        total_loss = 0
        timer = Timer(alpha=0.1)
        start_time = time.time()

        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch

            # Set epoch for distributed sampler/batch_sampler
            if hasattr(self.train_dataloader, "batch_sampler"):
                batch_sampler = self.train_dataloader.batch_sampler
                if hasattr(batch_sampler, "set_epoch"):
                    batch_sampler.set_epoch(epoch)
            elif hasattr(self.train_dataloader, "sampler"):
                sampler = self.train_dataloader.sampler
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)

            for batch in self.train_dataloader:

                with self.accelerator.accumulate(self.model):

                    # Forward pass
                    with self.accelerator.autocast():
                        fwd_start = time.time()
                        loss, additional_metrics = self._compute_loss(batch)
                        timer.update_fwd(time.time() - fwd_start)

                    # Backward pass
                    bwd_start = time.time()
                    self.accelerator.backward(loss)
                    timer.update_bwd(time.time() - bwd_start)

                    # Gradient clipping
                    if self.config.max_grad_norm is not None:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                    else:
                        grad_norm = 0

                    # Optimizer step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    # torch.cuda.empty_cache()

                # Accumulate loss (only log on accumulation boundary)
                total_loss += loss.detach().float()

                # Only log/save on accumulation boundaries
                if self.accelerator.sync_gradients:

                    # Logging
                    if self.global_step % self.config.logging_steps == 0:

                        allocated = torch.cuda.memory_allocated() / 1e9
                        max_allocated = torch.cuda.max_memory_allocated() / 1e9
                        reserved = torch.cuda.memory_reserved() / 1e9

                        if self.global_step % 100 == 0:
                            torch.cuda.reset_peak_memory_stats()

                        current_time = time.time()
                        elapsed = current_time - start_time
                        steps_per_sec = (
                            self.config.logging_steps / elapsed if elapsed > 0 else 0
                        )

                        # Average loss over accumulation steps
                        avg_loss = total_loss / self.config.gradient_accumulation_steps

                        metrics = {
                            "train_loss": avg_loss.item(),
                            "learning_rate": self.lr_scheduler.get_last_lr()[0],
                            "grad_norm": (
                                grad_norm.item()
                                if hasattr(grad_norm, "item")
                                else grad_norm
                            ),
                            "allocated_memory": allocated,
                            "max_allocated_memory": max_allocated,
                            "reserved_memory": reserved,
                            "steps_per_sec": steps_per_sec,
                            "forward_time": timer.fwd_avg,
                            "backward_time": timer.bwd_avg,
                            "epoch": epoch,
                            **additional_metrics,
                        }

                        self._log_metrics(metrics, self.global_step)
                        start_time = current_time

                    total_loss = 0

                    # Evaluation
                    if (
                        self.eval_dataloader is not None
                        and self.global_step > 0
                        and self.config.eval_steps > 0
                        and self.global_step % self.config.eval_steps == 0
                    ):
                        logging.info("Beginning evaluation...")

                        eval_metrics = self.eval_fn(self.eval_dataloader)
                        eval_metrics = {
                            f"eval_{k}" if not k.startswith("eval_") else k: v
                            for k, v in eval_metrics.items()
                        }
                        self._log_metrics(eval_metrics, self.global_step)

                        # Save best model
                        eval_loss = eval_metrics.get("eval_loss", float("inf"))
                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self.save_checkpoint(self.global_step, is_best=True)

                        self.model.train()

                    # Save checkpoint
                    if (
                        self.global_step > 0
                        and self.global_step % self.config.save_steps == 0
                    ):
                        self.save_checkpoint(self.global_step)

                    self.global_step += 1

                    # Check if we've reached max steps
                    if (
                        self.config.max_steps > 0
                        and self.global_step >= self.config.max_steps
                    ):
                        break

            if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                break

        # Final checkpoint save
        if self.accelerator.is_main_process:
            self.save_checkpoint(self.global_step)

        # Call pre-end callback if defined (for custom final steps before wandb ends)
        if hasattr(self, "_pre_end_callback") and self._pre_end_callback:
            self._pre_end_callback()

        # End wandb run
        if (
            self.config.log_wandb
            and WANDB_AVAILABLE
            and self.accelerator.is_main_process
        ):
            self.accelerator.end_training()
