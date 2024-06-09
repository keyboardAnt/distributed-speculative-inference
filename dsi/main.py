"""The main entrypoint for the CLI."""

import os
import hydra
from omegaconf import OmegaConf

from dsi.config import Config

@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
