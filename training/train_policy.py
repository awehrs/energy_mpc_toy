import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict

import hydra
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset


from models.agent import ToolAgent
from models.policy import FlowPolicy
from models.energy import Energy, CompletionModel
from dataset.trajectory_dataset import TrajectoryDataset
from dataset.collation import BucketBatchSampler, TrajectoryCollator


logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="Removed shared tensor.*while saving")

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# State cache
# ---------------------------------------------------------------------------


class StateCache:
    """Flat CPU cache of JEPA memory states for energy/completion/policy training.

    Built once at init (JEPA is frozen post-pretraining). Stores all valid
    per-step states and their successors as flat tensors on CPU.

    Args:
        states:      [N, Ns, d]  — all valid states, CPU
        next_states: [N, Ns, d]  — successor state (zeros for terminal steps)
        has_next:    [N]  bool   — False for last step of each trajectory
        is_terminal: [N]  bool   — True for last n_terminal_steps of each trajectory
    """

    def __init__(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        has_next: torch.Tensor,
        is_terminal: torch.Tensor,
    ):
        self.states = states  # [N, Ns, d] CPU
        self.next_states = next_states  # [N, Ns, d] CPU
        self.has_next = has_next  # [N] bool CPU
        self.is_terminal = is_terminal  # [N] bool CPU

    @property
    def terminal(self) -> torch.Tensor:
        return self.states[self.is_terminal]

    @property
    def nonterminal(self) -> torch.Tensor:
        return self.states[~self.is_terminal]

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample n random states for policy rollout starting points. [n, Ns, d]"""
        idx = torch.randperm(len(self.states))[:n]
        return self.states[idx].to(device)

    def __len__(self):
        return len(self.states)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class AlternatingTrainer:
    """Post-JEPA trainer: completion (once) → (energy, policy) x N cycles.

    One Accelerator, two optimizers — energy_optimizer covers the energy head
    and s_terminal (plus completion model once it exists); policy_optimizer
    covers the flow policy. Each phase only steps the relevant optimizer.

    The static dataset (JEPA trajectories with terminal STOP steps) is used
    for completion and energy training. Policy training accepts an optional
    `make_dataloader` factory for a mixed static+rollout dataset; when None
    it falls back to the static dataloader.
    """

    def __init__(
        self,
        agent: ToolAgent,
        config: DictConfig,
        model_config: DictConfig,
        static_dataset: Dataset,
        collate_fn: Callable,
        batch_sampler_cls: Callable,
    ):
        self.config = config
        self.model_config = model_config
        self._unwrapped_agent = agent

        # ---- Accelerator ----
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=(
                config.mixed_precision if config.mixed_precision != "none" else None
            ),
            log_with="wandb" if config.log_wandb and WANDB_AVAILABLE else None,
            project_dir=config.output_dir,
        )
        self.accelerator.even_batches = config.even_batches
        set_seed(config.seed)

        if self.accelerator.is_main_process:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # ---- Optimizers ----
        energy_params = list(agent.energy.parameters()) + [agent.s_terminal]
        if agent.completion is not None:
            energy_params += list(agent.completion.parameters())

        opt_kwargs = dict(
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )
        self.energy_optimizer = torch.optim.AdamW(energy_params, **opt_kwargs)
        self.policy_optimizer = torch.optim.AdamW(
            list(agent.policy.parameters()), **opt_kwargs
        )

        # ---- Static dataloader ----
        batch_sampler = batch_sampler_cls(static_dataset)
        static_dataloader = DataLoader(
            static_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=config.dataloader_num_workers,
            pin_memory=torch.cuda.is_available() and config.dataloader_pin_memory,
        )

        # ---- LR schedulers ----
        from transformers import get_cosine_schedule_with_warmup

        steps_per_epoch = len(batch_sampler) // config.gradient_accumulation_steps
        total_steps = steps_per_epoch * config.num_epochs

        self.energy_scheduler = get_cosine_schedule_with_warmup(
            self.energy_optimizer, config.warmup_steps, total_steps
        )
        self.policy_scheduler = get_cosine_schedule_with_warmup(
            self.policy_optimizer, config.warmup_steps, total_steps
        )

        # ---- accelerator.prepare ----
        (
            self.agent,
            self.energy_optimizer,
            self.policy_optimizer,
            self.static_dataloader,
            self.energy_scheduler,
            self.policy_scheduler,
        ) = self.accelerator.prepare(
            agent,
            self.energy_optimizer,
            self.policy_optimizer,
            static_dataloader,
            self.energy_scheduler,
            self.policy_scheduler,
        )

        if config.log_wandb and WANDB_AVAILABLE and self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=config.wandb_project,
                config=OmegaConf.to_object(config),
            )

        self.cycle = 0
        self.global_step = 0

        logger.info("Building state cache (frozen JEPA forward)...")
        self.state_cache = self._build_state_cache(
            n_terminal_steps=config.n_terminal_steps
        )
        logger.info(
            f"State cache built: {len(self.state_cache)} steps "
            f"({self.state_cache.is_terminal.sum().item()} terminal)"
        )

    # ------------------------------------------------------------------
    # State cache
    # ------------------------------------------------------------------

    def _build_state_cache(self, n_terminal_steps: int = 2) -> StateCache:
        """Run frozen JEPA forward over entire static dataset, cache memory states.

        Args:
            n_terminal_steps: number of final steps per trajectory marked as terminal
                              (label=1 for completion model). Using >1 prevents the
                              model from cheating by memorising a single exact state.
        Returns:
            StateCache with all valid states on CPU.
        """
        self.agent.eval()
        all_states: list[torch.Tensor] = []
        all_next_states: list[torch.Tensor] = []
        all_has_next: list[torch.Tensor] = []
        all_is_terminal: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in self.static_dataloader:
                with self.accelerator.autocast():
                    outputs = self._unwrapped_agent.jepa_forward(
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
                        targets=None,
                    )

                states = outputs["state"]  # [B, T, Ns, d]
                cu_traj_lens = batch["cu_traj_lens"]  # [B+1] step-level
                B = states.shape[0]

                # Process trajectory by trajectory to build consecutive pairs
                for i in range(B):
                    traj_len = int((cu_traj_lens[i + 1] - cu_traj_lens[i]).item())
                    s = states[i, :traj_len].cpu()  # [T_i, Ns, d]

                    # next_states: shift by 1; last step has no successor
                    next_s = torch.zeros_like(s)
                    next_s[:-1] = s[1:]
                    has_next = torch.ones(traj_len, dtype=torch.bool)
                    has_next[-1] = False

                    # terminal: last n_terminal_steps
                    is_term = torch.zeros(traj_len, dtype=torch.bool)
                    is_term[max(0, traj_len - n_terminal_steps) :] = True

                    all_states.append(s)
                    all_next_states.append(next_s)
                    all_has_next.append(has_next)
                    all_is_terminal.append(is_term)

        return StateCache(
            states=torch.cat(all_states, dim=0),
            next_states=torch.cat(all_next_states, dim=0),
            has_next=torch.cat(all_has_next, dim=0),
            is_terminal=torch.cat(all_is_terminal, dim=0),
        )

    # ------------------------------------------------------------------
    # Training phases
    # ------------------------------------------------------------------

    def run_completion_phase(self):
        """Train completion model (once) via BCE on terminal vs non-terminal states.

        Iterates over terminal states (minority), pairing each mini-batch with an
        equal-sized sample of non-terminal states for class balance.
        Uses energy_optimizer (completion model params are in its param group).
        """
        logger.info("Completion phase start")
        self.agent.train()

        device = self.accelerator.device
        mini_batch = self.config.completion_batch_size
        terminal = self.state_cache.terminal  # [Nt, Ns, d]
        nonterminal = self.state_cache.nonterminal  # [Nnt, Ns, d]

        n_epochs = self.config.completion_epochs
        logging_steps = self.config.logging_steps
        running_loss, num_steps = 0.0, 0

        for _epoch in range(n_epochs):
            perm = torch.randperm(len(terminal))

            for start in range(0, len(terminal), mini_batch):
                t_batch = terminal[perm[start : start + mini_batch]].to(device)
                nt_idx = torch.randint(len(nonterminal), (len(t_batch),))
                nt_batch = nonterminal[nt_idx].to(device)

                with self.accelerator.accumulate(self.agent):
                    with self.accelerator.autocast():
                        loss = self.agent.train_completion_model(t_batch, nt_batch)

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.energy_optimizer.param_groups[0]["params"],
                            self.config.max_grad_norm,
                        )
                        self.energy_optimizer.step()
                        self.energy_scheduler.step()
                        self.energy_optimizer.zero_grad()
                        running_loss += loss.detach().item()
                        num_steps += 1
                        self.global_step += 1

                        if num_steps % logging_steps == 0:
                            self._log({"completion_loss": running_loss / logging_steps})
                            running_loss = 0.0

        logger.info("Completion phase done")

        # Freeze completion model — not updated again after this phase
        for p in self.agent.completion.parameters():
            p.requires_grad_(False)

    def _policy_weights(self) -> tuple[float, float]:
        """Linear ramp: dataset_weight 1→(1-max_policy_weight), policy_weight 0→max_policy_weight."""
        n_cycles = self.config.n_cycles
        max_pw = self.config.max_policy_weight
        policy_weight = (
            max_pw * (self.cycle / max(n_cycles - 1, 1)) if n_cycles > 1 else max_pw
        )
        return 1.0 - policy_weight, policy_weight

    def run_energy_epoch(self):
        """One epoch updating the energy model on the state cache."""
        logger.info(f"[cycle {self.cycle}] Energy epoch start")
        self.agent.train()

        device = self.accelerator.device
        mini_batch = self.config.energy_batch_size
        cache = self.state_cache
        dataset_weight, policy_weight = self._policy_weights()
        n_policy_samples = self.config.n_policy_samples if policy_weight > 0.0 else 0
        logger.info(
            f"[cycle {self.cycle}] dataset_weight={dataset_weight:.3f}, policy_weight={policy_weight:.3f}, n_policy_samples={n_policy_samples}"
        )

        logging_steps = self.config.logging_steps
        running_loss, num_steps = 0.0, 0
        perm = torch.randperm(len(cache))

        for start in range(0, len(cache), mini_batch):
            idx = perm[start : start + mini_batch]
            states = cache.states[idx].to(device)
            next_states = cache.next_states[idx].to(device)
            has_next = cache.has_next[idx].to(device)

            with self.accelerator.accumulate(self.agent):
                with self.accelerator.autocast():
                    loss = self.agent.train_energy_model(
                        states,
                        next_states,
                        has_next,
                        dataset_weight,
                        policy_weight,
                        n_policy_samples,
                    )

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.energy_optimizer.param_groups[0]["params"],
                        self.config.max_grad_norm,
                    )
                    self.energy_optimizer.step()
                    self.energy_scheduler.step()
                    self.energy_optimizer.zero_grad()
                    running_loss += loss.detach().item()
                    num_steps += 1
                    self.global_step += 1

                    if num_steps % logging_steps == 0:
                        self._log({"energy_loss": running_loss / logging_steps})
                        running_loss = 0.0

        logger.info(f"[cycle {self.cycle}] Energy epoch done")

    def run_policy_epoch(self):
        """One epoch updating the policy via differentiable rollout loss.

        Samples starting states from the cache, fans out K rollouts per state,
        and backprops through the predictor + energy to the policy params.
        """
        logger.info(f"[cycle {self.cycle}] Policy epoch start")
        self.agent.train()

        device = self.accelerator.device
        mini_batch = self.config.policy_batch_size
        horizon = self.model_config.plan.horizon
        num_samples = self.model_config.plan.num_samples
        n_steps = (
            len(self.state_cache) // mini_batch
            if self.config.policy_steps_per_epoch == -1
            else self.config.policy_steps_per_epoch
        )

        logging_steps = self.config.logging_steps
        running_loss, num_steps = 0.0, 0

        for _ in range(n_steps):
            states = self.state_cache.sample(mini_batch, device)  # [B, Ns, d]

            with self.accelerator.accumulate(self.agent):
                with self.accelerator.autocast():
                    loss = self.agent.policy_rollout_loss(states, horizon, num_samples)

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.policy_optimizer.param_groups[0]["params"],
                        self.config.max_grad_norm,
                    )
                    self.policy_optimizer.step()
                    self.policy_scheduler.step()
                    self.policy_optimizer.zero_grad()
                    running_loss += loss.detach().item()
                    num_steps += 1
                    self.global_step += 1

                    if num_steps % logging_steps == 0:
                        self._log({"policy_loss": running_loss / logging_steps})
                        running_loss = 0.0

        logger.info(f"[cycle {self.cycle}] Policy epoch done")

    # ------------------------------------------------------------------
    # Top-level loop
    # ------------------------------------------------------------------

    def train(self, n_cycles: int = 1):
        """Full post-JEPA training: completion (once) → (energy + policy) x n_cycles."""
        self.run_completion_phase()
        self._log(self.eval_completion())

        for cycle in range(n_cycles):
            self.cycle = cycle
            logger.info(f"--- Cycle {cycle + 1}/{n_cycles} ---")
            self.run_energy_epoch()
            self.run_policy_epoch()

            eval_metrics = {}
            eval_metrics.update(self.eval_energy())
            eval_metrics.update(self.eval_policy())
            self._log(eval_metrics)

            self._save_checkpoint(cycle)

        if (
            self.config.log_wandb
            and WANDB_AVAILABLE
            and self.accelerator.is_main_process
        ):
            self.accelerator.end_training()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def eval_completion(self, n_samples: int = 1024) -> Dict[str, float]:
        """Completion model accuracy on balanced terminal/nonterminal states."""
        self.agent.eval()
        device = self.accelerator.device
        cache = self.state_cache

        n = min(n_samples // 2, len(cache.terminal), len(cache.nonterminal))
        term = cache.terminal[torch.randperm(len(cache.terminal))[:n]].to(device)
        nonterm = cache.nonterminal[torch.randperm(len(cache.nonterminal))[:n]].to(
            device
        )

        states = torch.cat([term, nonterm])
        labels = torch.cat(
            [torch.ones(n, device=device), torch.zeros(n, device=device)]
        )

        probs = self.agent.completion.predict(states)
        preds = (probs > 0.5).float()
        acc = (preds == labels).float().mean().item()
        term_acc = (preds[:n] == 1).float().mean().item()
        nonterm_acc = (preds[n:] == 0).float().mean().item()

        self.agent.train()
        return {
            "eval/completion_acc": acc,
            "eval/completion_term_acc": term_acc,
            "eval/completion_nonterm_acc": nonterm_acc,
        }

    @torch.no_grad()
    def eval_energy(self, n_samples: int = 1024) -> Dict[str, float]:
        """Mean energy of terminal vs non-terminal states (terminal should be ~0)."""
        self.agent.eval()
        device = self.accelerator.device
        cache = self.state_cache

        n_term = min(n_samples // 2, len(cache.terminal))
        n_nonterm = min(n_samples // 2, len(cache.nonterminal))
        term = cache.terminal[torch.randperm(len(cache.terminal))[:n_term]].to(device)
        nonterm = cache.nonterminal[
            torch.randperm(len(cache.nonterminal))[:n_nonterm]
        ].to(device)

        energy_term = self.agent.energy(term).mean().item()
        energy_nonterm = self.agent.energy(nonterm).mean().item()

        self.agent.train()
        return {
            "eval/energy_terminal": energy_term,
            "eval/energy_nonterminal": energy_nonterm,
        }

    @torch.no_grad()
    def eval_policy(self, n_samples: int = 256) -> Dict[str, float]:
        """Mean energy of states reached by H-step policy rollouts."""
        self.agent.eval()
        energy = self.agent.policy_rollout_loss(
            self.state_cache.sample(n_samples, self.accelerator.device),
            horizon=self.model_config.plan.horizon,
            num_samples=self.model_config.plan.num_samples,
        ).item()
        self.agent.train()
        return {"eval/policy_rollout_energy": energy}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, metrics: Dict[str, Any]):
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            metrics = {**metrics, "gpu_mem_peak_gb": peak_gb}
            torch.cuda.reset_peak_memory_stats()
        log_str = ", ".join(f"{k}: {v:.6f}" for k, v in metrics.items())
        self.accelerator.print(f"Step {self.global_step}: {log_str}")
        if self.config.log_wandb and WANDB_AVAILABLE:
            self.accelerator.log(metrics, step=self.global_step)

    def _save_checkpoint(self, cycle: int):
        if not self.accelerator.is_main_process:
            return
        save_dir = Path(self.config.output_dir) / f"cycle-{cycle}"
        self._unwrapped_agent.save_components(
            save_dir,
            self._unwrapped_agent.config,
            components=["energy", "policy", "completion", "s_terminal"],
        )
        logger.info(f"Saved cycle-{cycle} checkpoint to {save_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(version_base="1.1", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    dataset = TrajectoryDataset.load(
        src_dir=Path(cfg.training.cache_dir, cfg.training.dataset_name)
    )

    collator = TrajectoryCollator(pad_token_id=dataset.tokenizer.pad_token_id)

    def make_batch_sampler(ds):
        return BucketBatchSampler(ds, max_tokens=cfg.training.max_tokens_per_batch)

    # Load JEPA-pretrained agent and attach energy + policy heads
    agent = ToolAgent.from_pretrained(
        load_dir=Path(cfg.training.jepa_checkpoint),
        config=cfg,
    )
    if agent.energy is None:
        agent.energy = Energy(
            d_model=cfg.model.d_model,
            n_heads=cfg.model.n_heads,
        )
    if agent.completion is None:
        agent.completion = CompletionModel(
            d_model=cfg.model.d_model,
            n_heads=cfg.model.n_heads,
        )
    if agent.policy is None:
        agent.policy = FlowPolicy(
            d_model=cfg.model.d_model,
            n_heads=cfg.model.n_heads,
            n_layers=cfg.model.policy.n_layers,
            n_action_latents=cfg.model.n_action_latents,
            time_emb_dim=cfg.model.policy.time_emb_dim,
            n_integration_steps=cfg.model.policy.n_integration_steps,
            norm_eps=cfg.model.norm_eps,
        )

    trainer = AlternatingTrainer(
        agent=agent,
        config=cfg.training,
        model_config=cfg.model,
        static_dataset=dataset,
        collate_fn=collator,
        batch_sampler_cls=make_batch_sampler,
    )

    trainer.train(n_cycles=cfg.training.n_cycles)
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
