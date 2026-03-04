import abc
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

from models.energy import Energy
from models.memory import Memory
from models.policy import Policy
from models.sensors import Sensor, LanguageSensor
from models.actuators import Actuator, LanguageActuator
from models.predictor import Predictor, JEPAPredictor
from models.memory import GatedDeltaMemory
from models.attention import Downsampler


class Agent(abc.ABC, nn.Module):
    pass


class ToolAgent(Agent):

    def __init__(
        self,
        config: DictConfig,
        energy: Optional[Energy] = None,
        memory: Optional[Memory] = None,
        policy: Optional[Policy] = None,
        sensor: Optional[Sensor] = None,
        actuator: Optional[Actuator] = None,
        predictor: Optional[Predictor] = None,
    ):
        super().__init__()
        self.config = config
        self.energy = energy
        self.memory = memory
        self.policy = policy
        self.sensor = sensor
        self.actuator = actuator
        self.predictor = predictor

        self.precept_downsampler = Downsampler(
            d_model=config.model.d_model,
            n_latents=config.model.n_precept_latents,
            n_heads=config.model.n_heads,
            norm_eps=config.model.norm_eps,
        )

        self.action_downsampler = Downsampler(
            d_model=config.model.d_model,
            n_latents=config.model.n_action_latents,
            n_heads=config.model.n_heads,
            norm_eps=config.model.norm_eps,
        )

        # Learned "no previous action" embedding for memory step 0
        self.null_action_latent = nn.Parameter(
            torch.randn(1, config.model.n_action_latents, config.model.d_model) * 0.02
        )

    def save_components(self, save_dir: Path, config: DictConfig) -> None:
        """Save each component's state_dict to separate files."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, save_dir / "config.yaml")

        if self.sensor is not None:
            torch.save(
                self.sensor.projection.state_dict(),
                save_dir / "sensor_projection.pt",
            )
        torch.save(
            self.precept_downsampler.state_dict(),
            save_dir / "precept_downsampler.pt",
        )
        torch.save(
            self.action_downsampler.state_dict(),
            save_dir / "action_downsampler.pt",
        )
        if self.memory is not None:
            torch.save(self.memory.state_dict(), save_dir / "memory.pt")
        if self.predictor is not None:
            torch.save(self.predictor.state_dict(), save_dir / "predictor.pt")
        if self.actuator is not None:
            torch.save(self.actuator.state_dict(), save_dir / "actuator.pt")
        if self.energy is not None:
            torch.save(self.energy.state_dict(), save_dir / "energy.pt")
        if self.policy is not None:
            torch.save(self.policy.state_dict(), save_dir / "policy.pt")
        torch.save(
            {"null_action_latent": self.null_action_latent.data},
            save_dir / "null_action_latent.pt",
        )

    @classmethod
    def from_pretrained(cls, load_dir: Path, config: DictConfig) -> "ToolAgent":
        """Construct agent and load component state dicts from a directory.

        Only loads components whose .pt files exist, so a partial checkpoint
        (e.g. JEPA-only, no actuator) works fine.
        """
        load_dir = Path(load_dir)

        sensor = LanguageSensor(
            model_name=config.model.sensor.model_name,
            d_model=config.model.d_model,
        )

        memory = GatedDeltaMemory(
            d_model=config.model.d_model,
            n_state_latents=config.model.n_precept_latents,
            n_heads=config.model.n_heads,
            n_blocks=config.model.memory.n_blocks,
            n_mixer_layers=config.model.memory.n_mixer_layers,
            norm_eps=config.model.norm_eps,
        )

        predictor = JEPAPredictor(
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            n_layers=config.model.predictor.n_layers,
            norm_eps=config.model.norm_eps,
        )

        agent = cls(
            config=config,
            sensor=sensor,
            memory=memory,
            predictor=predictor,
        )

        # Load available component state dicts
        component_map = {
            "sensor_projection.pt": agent.sensor.projection,
            "precept_downsampler.pt": agent.precept_downsampler,
            "action_downsampler.pt": agent.action_downsampler,
            "memory.pt": agent.memory,
            "predictor.pt": agent.predictor,
        }

        for filename, module in component_map.items():
            path = load_dir / filename
            if path.exists():
                module.load_state_dict(torch.load(path, weights_only=True))

        null_path = load_dir / "null_action_latent.pt"
        if null_path.exists():
            data = torch.load(null_path, weights_only=True)
            agent.null_action_latent.data.copy_(data["null_action_latent"])

        if (load_dir / "actuator.pt").exists():
            actuator = LanguageActuator(
                model_name=config.model.sensor.model_name,
                n_latents=config.model.n_action_latents,
                latent_dim=config.model.d_model,
                max_action_len=config.max_action_len,
                n_self_attn_layers=config.model.action_model.n_decoder_self_attn_layers,
                n_self_attn_heads=config.n_self_attn_heads,
                n_cross_attn_heads=config.n_cross_attn_heads,
                dropout=config.dropout,
            )
            actuator.load_state_dict(
                torch.load(load_dir / "actuator.pt", weights_only=True)
            )
            agent.actuator = actuator

        return agent

    def _vicreg_loss(
        self,
        latents: torch.Tensor,
        traj_mask: torch.Tensor,
        var_weight: float = 1.0,
        cov_weight: float = 0.04,
        var_target: float = 1.0,
    ) -> tuple:
        """Variance + covariance regularization on latent representations.

        Args:
            latents:   [B, T, N, d] — student precept latents
            traj_mask: [B, T] — True for valid steps
        Returns:
            (var_loss, cov_loss) — scalar losses
        """
        # Gather valid latents → [M, d] where M = num valid latent vectors
        valid = latents[traj_mask]  # [M_steps, N, d]
        x = valid.reshape(-1, latents.shape[-1])  # [M_steps * N, d]

        # Variance: per-dimension std should stay above var_target
        std = x.std(dim=0)  # [d]
        var_loss = var_weight * torch.clamp(var_target - std, min=0).mean()

        # Covariance: off-diagonal elements of covariance matrix should be zero
        x_centered = x - x.mean(dim=0)
        cov = (x_centered.T @ x_centered) / max(x.shape[0] - 1, 1)  # [d, d]
        d = cov.shape[0]
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        cov_loss = cov_weight * off_diag / d

        return var_loss, cov_loss

    def _unpack_and_pad(
        self,
        packed_latents: torch.Tensor,
        n_latents: int,
        cu_traj_lens: torch.Tensor,
        bsz: int,
        max_traj_len: int,
    ) -> torch.Tensor:
        """Reshape packed sensor output into padded per-trajectory format.

        Args:
            packed_latents: [sum(traj_lens) * n_latents, d]  — flat sensor output
            n_latents:      number of latent vectors per step
            cu_traj_lens:   [B + 1] cumulative step counts
            bsz:            batch size
            max_traj_len:   max trajectory length in batch

        Returns:
            [B, max_traj_len, n_latents, d]  — zero-padded at the end of each trajectory
        """
        d = packed_latents.shape[-1]
        per_step = packed_latents.view(
            -1, n_latents, d
        )  # [sum(traj_lens), n_latents, d]

        padded = packed_latents.new_zeros(bsz, max_traj_len, n_latents, d)
        for i in range(bsz):
            start = cu_traj_lens[i]
            end = cu_traj_lens[i + 1]
            padded[i, : end - start] = per_step[start:end]

        return padded

    def jepa_forward(
        self,
        precept_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        precept_max_seq_len: int,
        precept_cu_seq_lens: torch.Tensor,
        precept_position_ids: torch.Tensor,
        action_max_seq_len: int,
        action_cu_seq_lens: torch.Tensor,
        action_position_ids: torch.Tensor,
        max_traj_len: int,
        cu_traj_lens: torch.Tensor,
        traj_mask: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """JEPA forward pass: encode → fuse → remember → predict.

        Tokens arrive packed (flat, variable-length) for efficient flash-attention
        encoding, then get unpacked into padded per-trajectory tensors for the
        causal memory and predictor.

        Let:
            B  = batch size (number of trajectories)
            T  = max_traj_len (longest trajectory in batch, after padding)
            S  = sum of all step counts across trajectories (total steps)
            Np = n_precept_latents (precept downsampler query count, == Ns)
            Na = n_action_latents  (action downsampler query count)
            Ns = n_state_latents   (memory fusion query count, == Np)
            d  = model hidden dim

        Args:
            precept_tokens:       [total_precept_tokens]  — packed token ids
            action_tokens:        [total_action_tokens]   — packed token ids
            precept_max_seq_len:  int  — longest individual precept sequence
            precept_cu_seq_lens:  [S + 1]  — cumulative token counts per step
            precept_position_ids: [total_precept_tokens]  — per-token position ids
            action_max_seq_len:   int  — longest individual action sequence
            action_cu_seq_lens:   [S + 1]  — cumulative token counts per step
            action_position_ids:  [total_action_tokens]  — per-token position ids
            max_traj_len:         int  — T
            cu_traj_lens:         [B + 1]  — cumulative step counts per trajectory
            traj_mask:            [B, T]  — True for real steps, False for padding
            targets:              [B, T, Np, d]  — EMA encoder precept latents (optional)

        Pipeline:
            1. Sensor (frozen encoder):
               tokens → LLM → projection → [total_tokens, d]
            2. Downsample (packed flash cross-attn, separate heads):
               precept hidden [total_precept_tokens, d] → [S * Np, d]
               action  hidden [total_action_tokens, d]  → [S * Na, d]
            3. Unpack + pad:
               [S * Np, d] → [B, T, Np, d]
               [S * Na, d] → [B, T, Na, d]
            4. Memory (Perceiver fusion + GDN blocks):
               precepts [B, T, Np, d] + actions [B, T, Na, d] → state [B, T, Ns, d]
            5. Predictor (banded flex-attn):
               cat(state, action) [B, T, Ns+Na, d] → [B, T, Ns+Na, d]
               slice state portion → prediction [B, T, Ns, d]
            6. Loss (if targets provided):
               smooth_l1(prediction[:, :-1], targets[:, 1:]) masked by traj_mask

        Returns:
            Dict with:
                prediction: [B, T, Ns, d]  — predicted next-step latents
                loss:       scalar          — JEPA loss (only if targets provided)
        """

        bsz = len(cu_traj_lens) - 1
        n_precept_latents = self.precept_downsampler.query.shape[0]
        n_action_latents = self.action_downsampler.query.shape[0]
        n_state_latents = self.memory.n_state_latents

        assert (
            n_precept_latents == n_state_latents
        ), f"Np ({n_precept_latents}) must equal Ns ({n_state_latents}) for direct loss"

        # --- 1. Encode  ---

        t0 = time.time()
        precept_hidden = self.sensor(
            input_ids=precept_tokens,
            position_ids=precept_position_ids,
        )
        torch.cuda.synchronize()
        logger.info(f"    sensor(precept): {time.time() - t0:.2f}s")

        t0 = time.time()
        action_hidden = self.sensor(
            input_ids=action_tokens,
            position_ids=action_position_ids,
        )
        torch.cuda.synchronize()
        logger.info(f"    sensor(action): {time.time() - t0:.2f}s")

        # --- 2. Downsample  ---

        t0 = time.time()
        precept_latents = self.precept_downsampler(
            embedding=precept_hidden,
            max_seq_len=precept_max_seq_len,
            cu_seq_lens=precept_cu_seq_lens,
        )

        action_latents = self.action_downsampler(
            embedding=action_hidden,
            max_seq_len=action_max_seq_len,
            cu_seq_lens=action_cu_seq_lens,
        )
        torch.cuda.synchronize()
        logger.info(f"    downsamplers: {time.time() - t0:.2f}s")

        # --- 3. Change to batched format  ---

        precept_latents = self._unpack_and_pad(  # [B, T, Np, d]
            precept_latents,
            n_precept_latents,
            cu_traj_lens,
            bsz,
            max_traj_len,
        )

        action_latents = self._unpack_and_pad(  # [B, T, Na, d]
            action_latents,
            n_action_latents,
            cu_traj_lens,
            bsz,
            max_traj_len,
        )

        # --- 4. Memory: process trajectories ---
        # Shift actions by 1 at the latent level: memory sees previous action.
        # Step 0 gets a learned null embedding; step t>0 gets action_latents[t-1].

        null = self.null_action_latent.unsqueeze(0).expand(
            bsz, -1, -1, -1
        )  # [B, 1, Na, d]

        previous_action_latents = torch.cat(
            [null, action_latents[:, :-1]],
            dim=1,
        )  # [B, T, Na, d]

        t0 = time.time()
        state = self.memory(
            precepts=precept_latents,
            previous_action=previous_action_latents,
        )
        torch.cuda.synchronize()
        logger.info(f"    memory: {time.time() - t0:.2f}s")

        # --- 5. Predictor: predict next-step precept latents ---
        # Predictor sees current state + current action (unshifted).

        t0 = time.time()
        N = n_state_latents + n_action_latents
        block_mask = self.predictor.build_block_mask(traj_mask, N, state.device)
        torch.cuda.synchronize()
        logger.info(f"    block_mask: {time.time() - t0:.2f}s")

        t0 = time.time()
        pred_full = self.predictor(
            state=state,
            block_mask=block_mask,
            action=action_latents,
        )  # [B, T, Ns+Na, d]
        torch.cuda.synchronize()
        logger.info(f"    predictor: {time.time() - t0:.2f}s")

        # Slice out state portion as the prediction (drop action positions)
        prediction = pred_full[:, :, :n_state_latents, :]  # [B, T, Ns, d]

        # --- 6. Loss ---
        result = {"prediction": prediction}

        if targets is not None:
            # prediction[t] predicts targets[t+1]
            pred_shifted = prediction[:, :-1]  # [B, T-1, Ns, d]
            tgt_shifted = targets[:, 1:]  # [B, T-1, Np, d]

            # Mask: both source and target steps must be valid
            mask = traj_mask[:, :-1] & traj_mask[:, 1:]  # [B, T-1]
            mask = mask.unsqueeze(-1).unsqueeze(-1)  # [B, T-1, 1, 1]

            jepa_loss = nn.functional.smooth_l1_loss(
                pred_shifted,
                tgt_shifted,
                reduction="none",
            )
            n_elements = mask.sum() * n_state_latents * pred_shifted.shape[-1]
            jepa_loss = (jepa_loss * mask).sum() / n_elements.clamp(min=1)

            # VICReg on student precept latents
            vic_var, vic_cov = self._vicreg_loss(precept_latents, traj_mask)

            result["loss"] = jepa_loss + vic_var + vic_cov
            result["jepa_loss"] = jepa_loss.detach()
            result["vic_var"] = vic_var.detach()
            result["vic_cov"] = vic_cov.detach()

        return result

    def generate_policy_examples(
        self,
        prompts: Optional[torch.Tensor] = None,
        action_tokens: Optional[torch.Tensor] = None,
        precept_tokens: Optional[torch.Tensor] = None,
    ):
        if action_tokens is not None and precept_tokens is not None:
            # Training initial (behavior cloning) policy.

            # Encode action tokens and precepts

            # Downsample

            # Run memory model.

            # Actions = action latents

            # States = states
            pass

        else:
            # Improving inital policy
            assert prompts is not None

            # Roll out states from prompts

            # initial state = memory_model(sensor(prompt))

            # While i < last subsampled step + horizon steps:

            # Plan

            # Sample k noise vectors

            # k = k for initial step, k = 1 for subsequent

            # Run through policy model (in parallel)

            # Predict action conditioned next state

            # Slice out subsampled steps + horizon windows

            # For each slice (in parallel):

            # Score terminal energy = energy_student(terminal state)

            # Back prop to state noise seeds

            # Re run each slice with new noise seeds (one sample per prompt this time)

            # Return (state, action_latent) pairs

    def plan(
        self,
        state: torch.Tensor,
        horizion: int,
        num_samples: int,
    ) -> Tuple[torch.Tensor]:

        for i in range(horizion):
            # Sample actions.
            actions = self.policy(
                state,
                num_samples=num_samples if i == 0 else 1,
            )

            state = self.predictor.predict_multiple(state=state, actions=actions)

        energy = self.energy(state)

    def forward(
        self,
        precept: torch.Tensor,
        previous_action: Optional[torch.Tensor] = None,
    ):

        precept_latents = self.precept_downsampler(self.sensor(precept))

        state = self.memory.update(precept_latents, previous_action)

        actions = []

        energies = []

        for i in range(None):

            noise = None

            # repeat precepts to match noise
            action_latents = self.policy_model(state, noise)

            # Make predictions
            state_preds = self.predictor(state, action_latents)

            # Energies
            energies = self.energy_model(state_preds)

            # Winnow actions and precepts
            actions.append()

        # Get best trajectories
        return actions[:], energies[:]
