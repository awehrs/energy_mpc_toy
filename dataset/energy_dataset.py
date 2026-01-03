import os
import json
import shutil
import logging
import itertools
from pathlib import Path
from typing import Callable, Dict, Optional, Union, List, Tuple


import datasets
import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import concatenate_datasets, load_dataset
from einops import rearrange
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from dataset.trajectory_dataset import TrajectoryDataset


class EnergyDataset:
    def __init__(self):
        pass

    def __get_item__(self, idx):
        pass

    @classmethod
    def build_dataset(self, data_dir: Union[str, Path]):
        pass
