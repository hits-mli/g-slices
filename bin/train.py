"""Hydra entrypoint for TSFlow experiments."""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from gslice.experiment.runtime import run_training


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return run_training(config)


if __name__ == "__main__":
    main()
