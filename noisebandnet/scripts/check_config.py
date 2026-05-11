"""Compose and print a Hydra config from the repo's configs directory.

Usage:
  python -m noisebandnet.scripts.check_config paper
  python -m noisebandnet.scripts.check_config data/mfcc

This script prints the merged YAML to stdout.
"""

import sys
from omegaconf import OmegaConf
from hydra import initialize, compose, initialize_config_dir
from hydra.errors import HydraException
from importlib.resources import files

CONFIG_DIR = files("noisebandnet").joinpath("configs")


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 1:
        print("Usage: check_config <config_name>  (e.g. paper or ae/noiseband)")
        raise SystemExit(2)
    # Accept either 'paper' or 'ae/noiseband' or 'data/mfcc' etc.
    name = argv[0]
    # If user passed a top-level group like 'data/mfcc', compose expects that as config_name
    cfg_name = name if name.endswith(".yaml") else f"{name}.yaml"
    # Try using the repo-relative config path first (works when invoked from repo root)
    try:
        with initialize(
            config_path="noisebandnet/configs",
            job_name="check_config",
            version_base="1.1",
        ):
            cfg = compose(config_name=cfg_name)
    except HydraException:
        print("test")
        # Fallback: use the package resource absolute config dir (works when installed or different CWD)
        try:
            cfg_dir = str(CONFIG_DIR)
            with initialize_config_dir(
                config_dir=cfg_dir, job_name="check_config", version_base="1.1"
            ):
                cfg = compose(config_name=cfg_name)
        except Exception as e:
            print(f"Failed to compose config '{name}': {e}")
            raise
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
