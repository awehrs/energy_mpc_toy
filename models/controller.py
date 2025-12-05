from typing import Dict, Optional

import einops
from omegaconf import DictConfig
import torch
import torch.nn as nn


class MPC_Controller(nn.Module):

    def __init__(
        self,
        action_model: nn.Module,
        dynamics_model: nn.Module,
        energy_model: nn.Module,
        flow_model: nn.Module,
        actuator: nn.Module,
        sensor: nn.Module,
    ):
        self.action_model = action_model
        self.dynamics_model = dynamics_model
        self.energy_model = energy_model
        self.flow_model = flow_model
        self.actuator = actuator
        self.sensor = sensor

    def forward(self, query: torch.Tensor, attention_mask: torch.Tensor):

        precept_latents = self.sensor(query, attention_mask)

        precept_energy = self.energy_model(precept_latents)

        action_latents = None

        actions = []

        energies = []

        for i in range(10):

            noise = None

            # repeat precepts to match noise
            action_latents = self.flow_model(precept_latents, noise)

            # Make predictions
            precept_latents = self.dynamics_model(precept_latents, action_latents)

            # Winnow actions and precepts
            actions.append()

        # Get best trajectories
        return actions[:], energies[:]
