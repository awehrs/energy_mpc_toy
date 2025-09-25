"""
Energy MPC Trainer - extends AccelerateTrainer with specific loss computation and metrics.
"""

import os
import logging
import warnings
from typing import Dict
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from models.model import Model
from training.trainer import AccelerateTrainer
from retrieval.dataset import create_precomputed_dataset
from retrieval.build_dataset import build_dataset_pipeline

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="Removed shared tensor.*while saving")


class EnergyMPCTrainer(AccelerateTrainer):
    """Trainer for Energy-based MPC model with forward prediction and energy losses."""

    def _per_step_xent(
        self, logits: torch.Tensor, targets: torch.Tensor, attention_mask: torch.Tensor
    ):
        """
        logits:  [B, S, T, V]
        targets: [B, S, T]
        attention_mask: [B, S, T] - 1 for real tokens, 0 for padding
        returns per-step CE: [B, S]
        """
        B, S, T, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * S * T, V),
            targets.reshape(B * S * T),
            reduction="none",
        ).reshape(B, S, T)
        mask = attention_mask.float()
        denom = mask.sum(dim=-1).clamp_min(1.0)
        return (loss * mask).sum(dim=-1) / denom  # [B, S]

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute training losses for energy-based MPC model.

        Expected batch keys:
        - input_ids: [B, S, K, L]
        - retrieval_queries: [B, S, A, index_dim]
        - target_tokens: [B, S, T]
        """
        # Forward pass through model
        outputs = self.model(
            input_ids=batch["input_ids"],
            input_attention_mask=batch["attention_mask"],
            retrieval_queries=batch["retrieval_queries"],
            target_tokens=batch["target_tokens"],
            question_input_ids=batch["question_input_ids"],
            question_attention_mask=batch["question_attention_mask"],
        )

        targets = batch["target_tokens"]
        target_attention_mask = batch["target_attention_mask"]

        # Extract model outputs
        doc_latents_f = outputs["document_latents"]  # [B,S,Lf,D] (forward outputs)
        doc_latents_e = outputs["encoder_doc_latents"]  # [B,S,Le,D] (encoder outputs)
        logits = outputs["decoder_logits"]  # [B,S,T,V]
        energies = outputs["energies"]  # [B,S]

        B, S, _, D = doc_latents_f.shape

        # Loss computation parameters from config
        w_decode = getattr(self.config, "w_decode", 1.0)
        w_energy = getattr(self.config, "w_energy", 0.5)
        w_mono = getattr(self.config, "w_mono", 0.2)
        w_forward = getattr(self.config, "w_forward", 1.0)
        mono_margin = getattr(self.config, "mono_margin", 0.0)

        # 1) Decoding loss per step
        ce_per_step = self._per_step_xent(
            logits, targets, target_attention_mask
        )  # [B,S]
        loss_decode = ce_per_step.mean()

        # 2) Energy supervision toward decoding loss (lower CE -> lower energy, z-score normalized)
        with torch.no_grad():
            mu = ce_per_step.mean(dim=1, keepdim=True)
            sd = ce_per_step.std(dim=1, keepdim=True).clamp_min(1e-6)
            energy_tgt = (ce_per_step - mu) / sd  # [B,S]

        loss_energy = F.mse_loss(energies, energy_tgt)

        # 3) Monotonicity penalty (energy descent)
        if S > 1:
            e_t = energies[:, :-1]
            e_tp1 = energies[:, 1:]
            mono_violation = (e_tp1 - e_t + mono_margin).clamp_min(0.0)
            loss_mono = mono_violation.mean()
        else:
            loss_mono = energies.new_zeros(())

        # 4) Forward prediction: forward_docs[t] ≈ encoder_docs[t+1] (token-wise L1 loss)
        if S > 1:
            z_pred = doc_latents_f[:, :-1, :, :]  # [B,S-1,L,D]
            z_tgt = doc_latents_e[:, 1:, :, :]  # [B,S-1,L,D]
            loss_forward = F.l1_loss(z_pred, z_tgt)
        else:
            loss_forward = energies.new_zeros(())

        total = (
            w_decode * loss_decode
            + w_energy * loss_energy
            + w_mono * loss_mono
            + w_forward * loss_forward
        )

        # Store metrics for logging
        corr = energies.new_tensor(0.0)
        if (B * S) > 1:
            flat = torch.stack([energies.flatten(), ce_per_step.flatten()])
            corr = torch.corrcoef(flat)[0, 1].detach()

        # Compute latent diversity metrics for encoder outputs
        latent_metrics = self._compute_latent_diversity_metrics(doc_latents_e)

        # Compute detailed monotonicity metrics
        mono_metrics = self._compute_monotonicity_metrics(energies, ce_per_step)

        # Compute forward prediction quality metrics
        forward_metrics = {}
        if S > 1:
            forward_metrics = self._compute_forward_prediction_metrics(
                doc_latents_f[:, :-1, :, :], doc_latents_e[:, 1:, :, :]
            )

        self._last_batch_metrics = {
            # Core losses
            "loss/total": total.detach(),
            "loss/decode": loss_decode.detach(),
            "loss/energy": loss_energy.detach(),
            "loss/mono": loss_mono.detach(),
            "loss/forward": loss_forward.detach(),
            # Energy-decode correlation
            "metric/energy_decode_corr": corr,
            # Latent diversity (prevent mode collapse)
            **latent_metrics,
            # Monotonicity analysis
            **mono_metrics,
            # Forward prediction quality
            **forward_metrics,
            # Debug values
            "debug/mean_energy": energies.mean().detach(),
            "debug/mean_ce": ce_per_step.mean().detach(),
            "debug/energy_std": energies.std().detach(),
            "debug/ce_std": ce_per_step.std().detach(),
        }

        # Log sequence-level plots periodically for blog-ready visualizations
        plot_frequency = getattr(self.config, "plot_logging_frequency", 100)
        if hasattr(self, "global_step") and self.global_step % plot_frequency == 0:
            self._log_sequence_plots(ce_per_step, energies)

        return total

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

    def _compute_monotonicity_metrics(
        self, energies: torch.Tensor, ce_per_step: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute detailed monotonicity violation metrics.

        Args:
            energies: [B, S] energy values per step
            ce_per_step: [B, S] cross-entropy loss per step

        Returns:
            Dict of monotonicity metrics
        """
        B, S = energies.shape

        if S <= 1:
            return {
                "mono/violation_rate": energies.new_tensor(0.0),
                "mono/avg_violation_magnitude": energies.new_tensor(0.0),
                "mono/energy_descent_trend": energies.new_tensor(0.0),
                "mono/ce_descent_trend": energies.new_tensor(0.0),
            }

        # Energy monotonicity
        energy_diffs = energies[:, 1:] - energies[:, :-1]  # [B, S-1]
        energy_violations = (energy_diffs > 0).float()
        violation_rate = energy_violations.mean()
        violation_magnitude = energy_diffs.clamp_min(0.0).mean()

        # Overall energy trend (should be negative for descent)
        energy_trend = (energies[:, -1] - energies[:, 0]).mean()

        # Cross-entropy trend (should also be negative for improvement)
        ce_trend = (ce_per_step[:, -1] - ce_per_step[:, 0]).mean()

        # Correlation between energy and CE changes
        if B > 1:
            energy_change = energies[:, -1] - energies[:, 0]  # [B]
            ce_change = ce_per_step[:, -1] - ce_per_step[:, 0]  # [B]
            change_corr = torch.corrcoef(torch.stack([energy_change, ce_change]))[0, 1]
        else:
            change_corr = energies.new_tensor(0.0)

        return {
            "mono/violation_rate": violation_rate.detach(),
            "mono/avg_violation_magnitude": violation_magnitude.detach(),
            "mono/energy_descent_trend": energy_trend.detach(),
            "mono/ce_descent_trend": ce_trend.detach(),
            "mono/energy_ce_change_corr": change_corr.detach(),
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

    def _log_sequence_plots(self, ce_per_step: torch.Tensor, energies: torch.Tensor):
        """
        Log sequence-level CE vs Energy plots to W&B for blog-ready visualizations.

        Args:
            ce_per_step: [B, S] cross-entropy loss per step
            energies: [B, S] energy values per step
        """
        # Only log if wandb is enabled and we're the main process
        if not (self.config.log_wandb and self.accelerator.is_main_process):
            return

        import wandb

        B, S = ce_per_step.shape

        # Log line plots for first few examples to show sequence evolution
        max_examples = min(B, 3)  # Log a few examples to avoid clutter
        for b in range(max_examples):
            table = wandb.Table(columns=["step", "cross_entropy", "energy"])
            for s in range(S):
                table.add_data(s, ce_per_step[b, s].item(), energies[b, s].item())

            wandb.log(
                {
                    f"sequence/example_{b}_ce_vs_energy": wandb.plot.line(
                        table,
                        "step",
                        ["cross_entropy", "energy"],
                        title=f"Example {b}: CE vs Energy Across Steps",
                    )
                },
                step=self.global_step,
            )

        # Log correlation scatter plot across all steps and batches
        table = wandb.Table(columns=["cross_entropy", "energy", "step", "batch"])
        for b in range(B):
            for s in range(S):
                table.add_data(ce_per_step[b, s].item(), energies[b, s].item(), s, b)

        wandb.log(
            {
                "scatter/energy_vs_ce_all_steps": wandb.plot.scatter(
                    table,
                    "cross_entropy",
                    "energy",
                    title="Energy vs Cross-Entropy (All Steps & Batches)",
                )
            },
            step=self.global_step,
        )

        # Log step-wise distributions as histograms
        for s in range(min(S, 5)):  # First few steps only
            step_energies = energies[:, s].cpu().detach().numpy()
            step_ces = ce_per_step[:, s].cpu().detach().numpy()

            wandb.log(
                {
                    f"distributions/step_{s}_energy": wandb.Histogram(step_energies),
                    f"distributions/step_{s}_ce": wandb.Histogram(step_ces),
                },
                step=self.global_step,
            )


@hydra.main(version_base="1.1", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training entry point."""

    # Stay in original working directory instead of Hydra's output dir
    original_cwd = HydraConfig.get().runtime.cwd
    os.chdir(original_cwd)

    logger.info("Starting Energy MPC Training...")

    # Check if dataset files exist
    chunks_file_path = Path(cfg.dataset.chunks_file)
    examples_file_path = Path(cfg.dataset.examples_file)
    knn_cache_file_path = Path(cfg.dataset.knn_cache_file)
    query_vectors_file_path = Path(cfg.dataset.query_vectors_file)
    example_gold_sets_file_path = Path(cfg.dataset.example_gold_sets_file)

    if not (
        chunks_file_path.exists()
        and examples_file_path.exists()
        and knn_cache_file_path.exists()
        and query_vectors_file_path.exists()
        and example_gold_sets_file_path.exists()
    ):
        logger.info("Building dataset pipeline...")
        train_dataset = build_dataset_pipeline(cfg)
        logger.info("Dataset pipeline complete!")
    else:
        logger.info("Loading existing datasets...")
        train_dataset = create_precomputed_dataset(
            chunks_file=chunks_file_path,
            examples_file=examples_file_path,
            knn_cache_file=Path(cfg.dataset.knn_cache_file),
            query_vectors_file=Path(cfg.dataset.query_vectors_file),
            example_gold_sets_file=Path(cfg.dataset.example_gold_sets_file),
            n_docs=cfg.dataset.n_docs,
            max_steps=cfg.dataset.max_steps,
            index_dim=cfg.dataset.index_dim,
            random_seed=cfg.training.seed,
        )

        # Log gold coverage statistics
        examples_with_gold = sum(
            1
            for gold_queries in train_dataset.example_gold_sets.values()
            if len(gold_queries) > 0
        )
        total_examples = len(train_dataset.example_gold_sets)
        coverage_percent = (
            examples_with_gold / total_examples * 100 if total_examples > 0 else 0
        )
        logger.info(
            f"Training dataset gold coverage: {examples_with_gold}/{total_examples} examples ({coverage_percent:.1f}%)"
        )

        if coverage_percent < 100:
            logger.warning(
                f"Only {coverage_percent:.1f}% of examples have gold coverage - consider increasing n_queries in KNN cache"
            )

    # Build model
    logger.info("Building model...")
    model = Model(cfg.model)

    eval_dataset = None
    if hasattr(cfg, "eval_chunks_file") and cfg.eval_chunks_file:
        logger.info("Loading evaluation dataset...")
        eval_dataset = create_precomputed_dataset(
            chunks_file=cfg.eval_chunks_file,
            examples_file=cfg.eval_examples_file,
            knn_cache_file=Path(cfg.eval_knn_cache_file),
            query_vectors_file=Path(cfg.eval_query_vectors_file),
            example_gold_sets_file=Path(cfg.eval_example_gold_sets_file),
            n_docs=cfg.model.n_docs,
            max_steps=cfg.model.max_steps,
            index_dim=cfg.dataset.index_dim,
            random_seed=cfg.seed + 1,  # Different seed for eval
        )

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = EnergyMPCTrainer(
        model=model,
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
