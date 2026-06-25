"""Maintainable DCMGNN implementation."""

from .config import DatasetConfig, get_dataset_config
from .data import BehaviorDataset, load_behavior_dataset
from .model import DCMGNN

__all__ = [
    "BehaviorDataset",
    "DCMGNN",
    "DatasetConfig",
    "get_dataset_config",
    "load_behavior_dataset",
]
