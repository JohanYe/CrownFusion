"""Training utilities extracted from train.py"""

import torch
import os
from collections import OrderedDict
from pathlib import Path
import shutil


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Step the EMA model towards the current model."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


def load_ddp(vae, ckpt_path, dict_key="model_state_dict"):
    """Load DDP checkpoint with proper key handling"""
    state_dict = torch.load(ckpt_path, weights_only=False)[dict_key]
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value

    vae.load_state_dict(new_state_dict)


def chamfer_validation(out_samples, crown_pc, chamfer_loss, device):
    """Calculate chamfer loss for validation
    
    Args:
        out_samples: [B, N, 3] subsampled point clouds from OMAGE
        crown_pc: [B, N, 3] ground truth point clouds  
        chamfer_loss: ChamferLoss instance
        device: torch device
    """
    batch_chamf_loss = chamfer_loss(crown_pc, out_samples)
    return batch_chamf_loss


def copy_orig_data_files(val_batch, viz_output_path, local_batch_size):
    """Copy original crown and tooth mesh files to the visualization output directory."""
    orig_crown_path = val_batch.get("orig_crown_path", [])
    orig_dentition_mesh_path = val_batch.get("orig_dentition_mesh_path", [])

    os.makedirs(viz_output_path, exist_ok=True)
    for i in range(min(local_batch_size, len(orig_crown_path))):
        orig_crown_dest = Path(viz_output_path) / f"orig_crown_{i}.ply"
        orig_dentition_mesh_dest = Path(viz_output_path) / f"orig_dentition_mesh_{i}.ply"

        if (
            i < len(orig_crown_path)
            and orig_crown_path[i]
            and Path(orig_crown_path[i]).exists()
        ):
            shutil.copy(orig_crown_path[i], orig_crown_dest)
        if (
            i < len(orig_dentition_mesh_path)
            and orig_dentition_mesh_path[i]
            and Path(orig_dentition_mesh_path[i]).exists()
        ):
            shutil.copy(orig_dentition_mesh_path[i], orig_dentition_mesh_dest)


def log_losses(loss_dict, batch_size, running_loss_dict):
    """Log losses to running loss dictionary"""
    for key, value in loss_dict.items():
        if key not in running_loss_dict:
            running_loss_dict[key] = 0.0
        if torch.is_tensor(value):
            running_loss_dict[key] += value.item() * batch_size
        else:
            running_loss_dict[key] += value * batch_size


