"""Checkpoint management utilities"""

import torch
import os


def save_checkpoint(checkpoint_data, checkpoint_dir, name_suffix):
    """Save checkpoint to directory"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{name_suffix}.pt")
    torch.save(checkpoint_data, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
    return checkpoint_path


def load_sit_checkpoint(cfg, model, ema, opt, ckpt_path, device, scaler=None):
    """Load SiT checkpoint with proper device handling"""
    print(f"Loading checkpoint from {ckpt_path}")

    # Convert device to proper format for map_location
    if isinstance(device, int):
        map_location = f"cuda:{device}"
    elif isinstance(device, torch.device):
        map_location = device
    else:
        map_location = device

    full_state_dict = torch.load(
        ckpt_path, map_location=map_location, weights_only=False
    )

    model.load_state_dict(full_state_dict["model"])
    ema.load_state_dict(full_state_dict["ema"])
    opt.load_state_dict(full_state_dict["opt"])

    # Load scaler state if available
    if (
        scaler is not None
        and "scaler" in full_state_dict
        and full_state_dict["scaler"] is not None
    ):
        scaler.load_state_dict(full_state_dict["scaler"])

    # Load args if not in debug mode
    if not cfg.debug.enabled and "args" in full_state_dict:
        cfg = full_state_dict["args"]

    # Move parameters to device
    for param in ema.parameters():
        param.data = param.data.to(device)

    for param in model.parameters():
        param.data = param.data.to(device)

    # Move optimizer state to device
    for state in opt.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    return cfg, scaler


def save_sit_checkpoint(
    model, ema, optimizer, cfg, scaler, checkpoint_dir, train_steps
):
    """Save SiT-specific checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{train_steps:07d}.pt")

    checkpoint_data = {
        "model": (
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        ),
        "ema": ema.state_dict(),
        "opt": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": cfg,
    }

    torch.save(checkpoint_data, checkpoint_path)
    return checkpoint_path
