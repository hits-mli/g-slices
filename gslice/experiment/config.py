"""Config utilities for Hydra experiment definitions.

The training code still expects plain dictionaries with the historical top-level
keys. These helpers keep that adapter explicit without coupling the training
entrypoints to Hydra internals.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

try:
    from omegaconf import DictConfig, OmegaConf
except ModuleNotFoundError:  # pragma: no cover - Hydra is optional for direct legacy entrypoints.
    DictConfig = None
    OmegaConf = None


TRAINING_KEYS = {
    "model",
    "setting",
    "model_params",
    "dataset_params",
    "trainer_params",
    "evaluation_params",
    "logdir",
    "seed",
}


def to_plain_dict(config: Any) -> dict[str, Any]:
    """Convert OmegaConf or mapping configs into a resolved plain dictionary."""
    if OmegaConf is not None and isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)
    if isinstance(config, Mapping):
        return copy.deepcopy(dict(config))
    raise TypeError(f"Expected mapping-like config, got {type(config).__name__}.")


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries without mutating either input."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def to_training_config(
    config: Any,
    *,
    include_runtime_logdir: bool = True,
    include_resolved_config: bool = False,
) -> dict[str, Any]:
    """Extract kwargs accepted by the historical train script ``main`` functions."""
    plain = to_plain_dict(config)
    training = {key: copy.deepcopy(plain[key]) for key in TRAINING_KEYS if key in plain}

    runtime = plain.get("runtime", {})
    if include_runtime_logdir and "logdir" not in training and isinstance(runtime, Mapping):
        logdir = runtime.get("logdir")
        if logdir is not None:
            training["logdir"] = logdir

    if include_resolved_config:
        config_snapshot = copy.deepcopy(training)
        training["config"] = config_snapshot

    return training
