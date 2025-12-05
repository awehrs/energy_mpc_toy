import os
import logging
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional


import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from training.trainer import AccelerateTrainer
from models.dynamics_model import DynamicsModel
from models.action_vae import VariationalAutoEncoder
from dataset.trajectory_dataset import TrajectoryDataset


logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="Removed shared tensor.*while saving")


class EMA_Encoder(nn.Module):
    def __init__(self, model: nn.Module, decay: float):
        super().__init__()

        self.decay = decay
        self.shadow = {}
        self.backuop = {}

        # Create shodow copy of parameters.
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    self.decay * self.shadow[name] * (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average.clone()

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Load EMA weights into the model."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])


class DynamicsTrainer(AccelerateTrainer):
    """Trainer for Energy-based MPC model with forward prediction and energy losses."""

    def __init__(
        self,
        dynamics_model: nn.Module,
        action_encoder: nn.Module,
        ema_encoder: nn.Module,
        config: DictConfig,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        eval_fn: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
    ):

        super().__init__(
            model=dynamics_model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_fn=eval_fn,
            tokenizer=tokenizer,
        )

        self.action_encoder = action_encoder

        if self.accelerator.is_main_process:
            self.ema_encoder = dynamics_model.sensor
        else:
            self.ema_encoder = None

    def teacher_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        if self.accelerator.is_main_process:
            with torch.no_grad():
                latents = self.ema_encoder(input_ids, attention_mask)
        else:
            latents = None

        latents = self.accelerator.broadcast(latents, from_process=0)

        return latents

    @torch.no_grad
    def update_ema(self) -> None:

        if not self.accelerator.is_main_process:
            return

        for e, o in zip(
            self.ema_encoder.parameters(), self.dynamics_model.sensor.parameters()
        ):
            e.data.mul_(self.decay).add_((1.0 - self.decay) * o.data)

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Sample actions
        action_params = self.action_encoder(input_ids, attention_mask)

        mean = action_params["mean"]
        log_var = action_params["log_var"]

        mean = torch.clamp(mean, min=-10, max=10)
        log_var = torch.clamp(log_var, min=-10, max=10)

        epsilon = torch.randn_like(mean)
        sigma = (0.5 * log_var).exp()

        action_latents = mean + sigma * epsilon

        # Forward pass.
        outputs = self.model(input_ids, attention_mask, action_latents)["output"][
            :, :-1, :
        ]

        targets = self.ema_encoder(input_ids, attention_mask)[:, 1:, :]

        loss_forward = F.l1_loss(outputs, targets)

        # Compute latent diversity metrics for encoder outputs
        latent_metrics = self._compute_latent_diversity_metrics()

        # Compute forward predict

    def _compute_latent_diversity_metrics(
        self, encoder_latents: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute step-level diversity metrics to track knowledge state evolution.

        Args:
            encoder_latents: [B, S, L, D] encoder document latents

        Returns:
            Dict of step-level diversity metrics
        """
        B, S, L, D = encoder_latents.shape

        # Pool to step-level knowledge states (what we actually care about)
        step_states = encoder_latents.mean(dim=2)  # [B, S, D]

        # 1. Overall step representation diversity
        all_steps = step_states.reshape(-1, D)  # [B*S, D]
        step_variance = all_steps.var(dim=0).mean()

        # 2. Evolution dynamics: how much do knowledge states change between steps?
        if S > 1:
            step_changes = step_states[:, 1:] - step_states[:, :-1]  # [B, S-1, D]
            evolution_magnitude = step_changes.norm(
                dim=-1
            ).mean()  # Average L2 change per step
            evolution_consistency = step_changes.norm(
                dim=-1
            ).std()  # Consistency of evolution

            # 3. Trajectory diversity: are different examples evolving differently?
            initial_states = step_states[:, 0]  # [B, D]
            final_states = step_states[:, -1]  # [B, D]
            initial_diversity = initial_states.var(dim=0).mean()
            final_diversity = final_states.var(dim=0).mean()
            trajectory_span = (final_states - initial_states).norm(dim=-1).mean()
        else:
            evolution_magnitude = torch.tensor(0.0, device=encoder_latents.device)
            evolution_consistency = torch.tensor(0.0, device=encoder_latents.device)
            initial_diversity = step_variance  # Same as overall when S=1
            final_diversity = step_variance
            trajectory_span = torch.tensor(0.0, device=encoder_latents.device)

        return {
            "diversity/step_variance": step_variance.detach(),
            "diversity/evolution_magnitude": evolution_magnitude.detach(),
            "diversity/evolution_consistency": evolution_consistency.detach(),
            "diversity/initial_diversity": initial_diversity.detach(),
            "diversity/final_diversity": final_diversity.detach(),
            "diversity/trajectory_span": trajectory_span.detach(),
        }

    def _compute_forward_prediction_metrics(
        self, z_pred: torch.Tensor, z_tgt: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute forward prediction quality metrics.

        Args:
            z_pred: [B, S-1, L, D] predicted latents from forward model
            z_tgt: [B, S-1, L, D] target latents from encoder

        Returns:
            Dict of forward prediction metrics
        """
        B, S_minus_1, L, D = z_pred.shape

        # Token-wise prediction accuracy
        token_mse = F.mse_loss(z_pred, z_tgt, reduction="none").mean(
            dim=-1
        )  # [B, S-1, L]
        avg_token_mse = token_mse.mean()

        # Cosine similarity per token
        z_pred_flat = z_pred.reshape(-1, D)
        z_tgt_flat = z_tgt.reshape(-1, D)
        cos_sim = F.cosine_similarity(z_pred_flat, z_tgt_flat, dim=-1)
        avg_cos_sim = cos_sim.mean()

        # Prediction consistency across steps
        if S_minus_1 > 1:
            step_consistency = F.mse_loss(
                token_mse[:, 1:], token_mse[:, :-1], reduction="mean"
            )
        else:
            step_consistency = token_mse.new_tensor(0.0)

        # Relative prediction error (normalized by target magnitude)
        target_norm = z_tgt.norm(dim=-1)  # [B, S-1, L]
        pred_error = (z_pred - z_tgt).norm(dim=-1)  # [B, S-1, L]
        relative_error = (pred_error / (target_norm + 1e-8)).mean()

        return {
            "forward/avg_token_mse": avg_token_mse.detach(),
            "forward/avg_cosine_similarity": avg_cos_sim.detach(),
            "forward/step_consistency": step_consistency.detach(),
            "forward/relative_error": relative_error.detach(),
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

    dataset = TrajectoryDataset.load(
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

    model_cfg = cfg.model.dynamics_model

    dynamics_model = DynamicsModel(
        model_name=model_cfg.model_name,
        d_action_latent=model_cfg.d_action_latent,
        d_ssm_model=model_cfg.d_ssm_model,
        n_precept_latents=model_cfg.n_precept_latents,
    )

    action_encoder = VariationalAutoEncoder.from_pretrained("checkpoint")

    trainer = DynamicsTrainer(
        dynamics_model=dynamics_model,
        action_encoder=action_encoder,
        config=cfg.training,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Start training
    logger.info("Starting training loop...")
    trainer.train()

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
