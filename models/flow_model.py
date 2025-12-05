from typing import Dict, Optional

import einops
from omegaconf import DictConfig
import torch
import torch.nn as nn


class FlowModel(nn.Module):

    def __init__(
        self,
        config: DictConfig,
        sensor: nn.Module,
    ):
        super().__init__()

        self.config = config

        self.sensor = sensor

        self.flow = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        mu, var = self.sensor(input_ids, attention_mask)["precept_latent"]

        precept_latent = None

        noise = None

        mu, log_var = self.flow(precept_latent, noise)

        return mu, log_var

    def generate(
        self,
        input_ids: torch.Tensor,
        input_attention_mask: torch.Tensor,
        retrieval_queries: torch.FloatTensor,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor = None,
        max_length: int = 512,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        pass
