"""Experiment configuration and runtime helpers."""

from gslice.experiment.config import (
    deep_merge,
    to_plain_dict,
    to_training_config,
)
from gslice.experiment.runtime import run_training

__all__ = [
    "deep_merge",
    "run_training",
    "to_plain_dict",
    "to_training_config",
]
