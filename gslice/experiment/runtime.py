"""Hydra runtime adapter for the existing training entrypoints."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

import yaml

from gslice.experiment.config import to_plain_dict, to_training_config


def _default_train_module(config: dict[str, Any]) -> str:
    model_name = str(config.get("model", ""))
    if model_name in {"lcde_gluonts", "lcde_gluonts_latent_fm"}:
        return "bin.train_lcde_gluonts"
    if model_name == "ps":
        return "bin.train_model_ps"
    return "bin.train_model"


def _write_resolved_config(config: dict[str, Any], logdir: str | None) -> None:
    if not logdir:
        return
    path = Path(logdir) / "resolved_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        yaml.safe_dump(config, fp, sort_keys=False)


def run_training(config: Any, *, loggers=None):
    """Run a Hydra/plain config through the legacy training backend."""
    plain = to_plain_dict(config)
    runtime = plain.get("runtime", {}) if isinstance(plain.get("runtime", {}), dict) else {}
    train_module_name = str(runtime.get("train_module") or _default_train_module(plain))
    train_module = importlib.import_module(train_module_name)

    train_kwargs = to_training_config(plain, include_runtime_logdir=True, include_resolved_config=True)
    if "logdir" in train_kwargs and hasattr(train_module, "_resolve_logdir"):
        train_kwargs["logdir"] = train_module._resolve_logdir(train_kwargs["logdir"])

    logdir = train_kwargs.get("logdir")
    if bool(runtime.get("write_resolved_config", True)):
        _write_resolved_config(plain, logdir)

    logging.info("Running %s with Hydra-resolved config.", train_module_name)
    return train_module.main(**train_kwargs, loggers=loggers)
