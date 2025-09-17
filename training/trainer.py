import os
import time
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs, TorchDynamoPlugin


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


class AccelerateTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        eval_fn: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.config = config
        self.model_config = model.config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.eval_fn = eval_fn or self._default_eval_fn
        self.tokenizer = tokenizer

        os.environ.setdefault("ACCELERATE_USE_DDP_FIND_UNUSED_PARAMETERS", "true")
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        # Initialize Accelerator with dynamo plugin if torch.compile is requested
        kwargs = {
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "mixed_precision": (
                config.mixed_precision if config.mixed_precision != "none" else None
            ),
            "log_with": "wandb" if config.log_wandb and WANDB_AVAILABLE else None,
            "project_dir": config.output_dir,
            "kwargs_handlers": [ddp_kwargs],
        }

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._param_counts = {"total": total_params, "trainable": trainable_params}

        if config.torch_compile:
            dynamo_plugin = TorchDynamoPlugin(
                backend="inductor",
                mode=config.torch_compile_mode,
                fullgraph=False,
                dynamic=True,
                disable_print_graph=True,
            )
            kwargs["dynamo_plugin"] = dynamo_plugin
            self.accelerator = Accelerator(**kwargs)
            self.accelerator.print(
                f"Using Accelerate dynamo backend with mode: {config.torch_compile_mode}"
            )
        else:
            self.accelerator = Accelerator(**kwargs)

        self.accelerator.print(f"Accelerator state: {self.accelerator.state}")
        self.accelerator.print(f"Mixed precision: {self.accelerator.mixed_precision}")
        self.accelerator.print(
            f"Gradient accumulation steps: {self.accelerator.gradient_accumulation_steps}"
        )

        # Set random seeds
        set_seed(config.seed)

        self.model = model

        # self.model = torch.compile(
        #     self.model,
        #     mode=config.torch_compile_mode,
        #     fullgraph=False,  # Important for complex models
        #     dynamic=True,  # Handle variable shapes better
        # )

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Create dataloaders
        train_dataloader = self._create_dataloader(self.train_dataset, is_train=True)
        eval_dataloader = None
        if self.eval_dataset is not None:
            eval_dataloader = self._create_dataloader(self.eval_dataset, is_train=False)

        # Calculate total training steps
        if config.max_steps > 0:
            self.total_steps = config.max_steps
        else:
            steps_per_epoch = (
                len(train_dataloader) // config.gradient_accumulation_steps
            )
            self.total_steps = steps_per_epoch * config.num_epochs

        # Initialize learning rate scheduler
        self.lr_scheduler = self._create_lr_scheduler(self.total_steps)

        # Prepare everything with Accelerator
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
                    loss = self._compute_loss(batch)

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

        return {"eval_loss": total_loss / num_samples}

    # MAKE this more generic in AccelerateTrainer:
    def _create_dataloader(self, dataset, is_train=True):
        """Create dataloader. Can be overridden for custom collation."""
        return DataLoader(
            dataset,
            batch_size=self._get_batch_size(is_train),
            shuffle=is_train,
            collate_fn=getattr(self, "collate_fn", None),  # Allow custom collation
            num_workers=self.config.dataloader_num_workers,
            pin_memory=(
                self.config.dataloader_pin_memory
                if torch.cuda.is_available()
                else False
            ),
            drop_last=is_train,
        )

    def _get_batch_size(self, is_train=True):
        """Get batch size for train/eval."""
        return (
            self.config.per_device_train_batch_size
            if is_train
            else self.config.per_device_eval_batch_size
        )

    def _log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log metrics to console and wandb."""
        # Console logging via accelerate (automatically handles main process)
        log_str = f"Step {step}: " + ", ".join(
            [f"{k}: {v:.6f}" for k, v in metrics.items()]
        )
        self.accelerator.print(log_str)

        # Add GPU metrics if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
            metrics.update(
                {
                    "gpu_memory_allocated_gb": gpu_memory,
                    "gpu_memory_cached_gb": gpu_memory_cached,
                }
            )

            # Add GPU utilization if GPUtil is available
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
        start_time = time.time()

        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch

            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    # Forward pass
                    with self.accelerator.autocast():
                        loss = self._compute_loss(batch)

                    # Backward pass
                    self.accelerator.backward(loss)

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

                # Accumulate loss (only log on accumulation boundary)
                total_loss += loss.detach().float()

                # Only log/save on accumulation boundaries
                if self.accelerator.sync_gradients:
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
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
                            "steps_per_sec": steps_per_sec,
                            "epoch": epoch,
                        }

                        # Add individual loss components if available (for subclasses)
                        if (
                            hasattr(self, "_last_batch_metrics")
                            and self._last_batch_metrics
                        ):
                            # Common metrics
                            common_metrics = {
                                "decode_loss": self._last_batch_metrics.get(
                                    "loss/decode", 0
                                ),
                                "energy_loss": self._last_batch_metrics.get(
                                    "loss/energy", 0
                                ),
                                "mono_loss": self._last_batch_metrics.get(
                                    "loss/mono", 0
                                ),
                                "forward_loss": self._last_batch_metrics.get(
                                    "loss/forward", 0
                                ),
                                # Add latent diversity metrics
                                "latent_variance": self._last_batch_metrics.get(
                                    "chunk_avg_variance",
                                    self._last_batch_metrics.get(
                                        "latent_avg_variance", 0
                                    ),
                                ),
                                "latent_distance": self._last_batch_metrics.get(
                                    "chunk_avg_distance",
                                    self._last_batch_metrics.get(
                                        "latent_avg_distance", 0
                                    ),
                                ),
                                "latent_participation": self._last_batch_metrics.get(
                                    "chunk_participation_ratio",
                                    self._last_batch_metrics.get(
                                        "latent_participation_ratio", 0
                                    ),
                                ),
                                # Add decode loss progression metrics
                                "decode_monotonic_pct": self._last_batch_metrics.get(
                                    "decode_monotonic_sequences", 0
                                ),
                                "decode_avg_step_drop": self._last_batch_metrics.get(
                                    "decode_avg_step_drop", 0
                                ),
                                "decode_total_drop": self._last_batch_metrics.get(
                                    "decode_total_drop", 0
                                ),
                                "decode_first_step": self._last_batch_metrics.get(
                                    "decode_first_step_loss", 0
                                ),
                                "decode_last_step": self._last_batch_metrics.get(
                                    "decode_last_step_loss", 0
                                ),
                            }

                            # Phase 1 specific metrics
                            if "monotonic_bonus" in self._last_batch_metrics:
                                common_metrics["monotonic_bonus"] = (
                                    self._last_batch_metrics.get("monotonic_bonus", 0)
                                )

                            # Full training specific metrics (energy/forward)
                            if "avg_energy_loss" in self._last_batch_metrics:
                                common_metrics.update(
                                    {
                                        "energy_loss": self._last_batch_metrics.get(
                                            "avg_energy_loss", 0
                                        ),
                                        "forward_loss": self._last_batch_metrics.get(
                                            "avg_forward_loss", 0
                                        ),
                                        "energy_weight": self._last_batch_metrics.get(
                                            "current_energy_weight", 0
                                        ),
                                        # Add energy-decode alignment metrics
                                        "energy_decode_corr": self._last_batch_metrics.get(
                                            "energy_decode_correlation", 0
                                        ),
                                        "energy_follows_pct": self._last_batch_metrics.get(
                                            "energy_follows_decode_pct", 0
                                        ),
                                        "decode_improvement": self._last_batch_metrics.get(
                                            "avg_decode_improvement", 0
                                        ),
                                        "energy_drop": self._last_batch_metrics.get(
                                            "avg_energy_drop", 0
                                        ),
                                    }
                                )

                            metrics.update(common_metrics)

                        self._log_metrics(metrics, self.global_step)
                        start_time = current_time

                    total_loss = 0

                    # Evaluation
                    if (
                        self.eval_dataloader is not None
                        and self.config.eval_steps > 0
                        and self.global_step % self.config.eval_steps == 0
                    ):

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
                        self.config.save_steps > 0
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
