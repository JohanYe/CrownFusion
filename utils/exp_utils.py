"""Experiment utilities for logging, directory management, and setup"""

import os
import logging
import random
from glob import glob
from omegaconf import OmegaConf
import torch
import numpy as np


def setup_seed(seed):
    """Setup random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_logger(logging_dir):
    """Create a logger that writes to a log file and stdout"""
    if logging_dir and os.path.exists(logging_dir):
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[\033[34m%(asctime)s\033[0m] %(message)s")
            )
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
    return logger


def create_experiment_dirs(
    results_dir, model_name, experiment_type="", debug_mode=False
):
    """Create experiment directories with consistent naming"""
    if debug_mode:
        return None, None, None, None

    os.makedirs(results_dir, exist_ok=True)

    # Find next experiment index
    existing_experiments = glob(f"{results_dir}/[0-9][0-9][0-9]-*")
    if existing_experiments:
        indices = [
            int(os.path.basename(exp).split("-")[0]) for exp in existing_experiments
        ]
        experiment_index = max(indices) + 1
    else:
        experiment_index = 0

    # Create experiment name
    model_string_name = model_name.replace("/", "-")
    random_suffix = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6))
    experiment_name = f"{experiment_index:03d}-{model_string_name}"
    if experiment_type:
        experiment_name += f"-{experiment_type}"
    experiment_name += f"-{random_suffix}"

    # Create directories
    experiment_dir = f"{results_dir}/{experiment_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    mesh_output_dir = f"{experiment_dir}/mesh_samples"

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(mesh_output_dir, exist_ok=True)

    return experiment_dir, checkpoint_dir, mesh_output_dir, experiment_name


def setup_wandb(cfg, experiment_name, project_name="default"):
    """Setup wandb logging"""
    if not cfg.get("wandb", {}).get("enabled", False):
        os.environ["WANDB_MODE"] = "disabled"
        import wandb

        wandb.init(mode="disabled")
        return

    try:
        import wandb
        import sit_utils.wandb_utils as wandb_utils

        wandb_key = os.environ.get("WANDB_KEY", "")
        if wandb_key:
            wandb.login(key=wandb_key)
        else:
            print(
                "WANDB_KEY is not set; skipping explicit wandb.login(). "
                "Run `export WANDB_KEY=...` or `wandb login` to authenticate."
            )
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=project_name,
            name=experiment_name,
            config=cfg_dict,
            id=wandb_utils.generate_run_id(experiment_name),
            resume="allow",
        )
    except ImportError:
        print("wandb not available, logging disabled")
        os.environ["WANDB_MODE"] = "disabled"


def save_config(cfg, experiment_dir, filename="config.yaml"):
    """Save configuration to experiment directory"""
    if experiment_dir:
        config_save_path = os.path.join(experiment_dir, filename)
        OmegaConf.save(cfg, config_save_path)
        try:
            import wandb

            if wandb.run is not None:
                wandb.save(config_save_path)
        except ImportError:
            pass
        return config_save_path
    return None
