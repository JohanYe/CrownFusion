# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is derived from SiT / DiT and is licensed under the MIT
# license found in sit_utils/LICENSE.txt.

import os
import fpsample
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
from omegaconf import OmegaConf
from copy import deepcopy
from time import time
import argparse
import numpy as np
from tqdm import tqdm

# Import utilities
from utils.env_utils import setup_environment
from utils.ddp_utils import (
    setup_distributed,
    cleanup_distributed,
    get_device_id,
    is_main_process,
)
from utils.exp_utils import (
    create_experiment_dirs,
    setup_wandb,
    save_config,
    create_logger,
    setup_seed,
)
from utils.checkpoint_utils import (
    load_sit_checkpoint,
    save_sit_checkpoint,
)

# Import project modules
import omages_utils
import sit_utils.wandb_utils as wandb_utils
from dataset import Conditional_Generation, PreEncodedConditionalGeneration
from sit_utils.models import AutoEncoder, SiT_models
from transport import Sampler, create_transport

from utils.training_utils import (
    update_ema,
    requires_grad,
    load_ddp,
    chamfer_validation,
    copy_orig_data_files,
)

#################################################################################
#                             Configuration Loading                             #
#################################################################################


def load_config(config_path, args_override=None):
    """Load configuration from YAML file and optionally override with command line args"""
    cfg = OmegaConf.load(config_path)

    # Override config with command line arguments if provided
    if args_override:
        override_mapping = {
            "data_path": ("data", "data_path"),
            "vae_ckpt": ("vae", "ckpt_path"),
            "results_dir": ("logging", "results_dir"),
            "wandb": ("wandb", "enabled"),
            "debug": ("debug", "enabled"),
            "use_pre_encoded": ("data", "use_pre_encoded"),
            "encoded_data_path": ("data", "encoded_data_path"),
            "orig_data_path": ("data", "orig_data_path"),
            "ckpt": ("resume", "ckpt_path"),
            "experiment_name": ("wandb", "experiment_name"),
        }

        for key, value in vars(args_override).items():
            if value is not None and key in override_mapping:
                section, param = override_mapping[key]
                setattr(cfg[section], param, value)

    # Apply debug mode overrides if enabled
    if cfg.debug.enabled:
        cfg.training.epochs = cfg.debug.epochs
        cfg.logging.sample_every = cfg.debug.sample_every
        cfg.logging.log_every = cfg.debug.log_every
        cfg.logging.ckpt_every = cfg.debug.ckpt_every
        cfg.logging.chamfer_every = cfg.debug.chamfer_every
        cfg.training.global_batch_size = cfg.debug.global_batch_size
        cfg.wandb.enabled = False
        cfg.training.num_workers = cfg.debug.num_workers

    return cfg


#################################################################################
#                                  Training Loop                                #
#################################################################################


def train_epoch(
    epoch,
    model,
    ema,
    opt,
    train_loader,
    vae,
    transport,
    transport_sampler,
    val_loader,
    cfg,
    device,
    scaler,
    logger,
    train_steps,
    running_loss,
    start_time,
    zs,
    model_fn,
    chamfer_loss,
    n_chamfer_batches,
    local_batch_size,
    mesh_output_dir,
    val_dataset,
    checkpoint_dir,
):
    """Train for one epoch"""
    model.train()
    gradient_accumulation_steps = getattr(
        cfg.training, "gradient_accumulation_steps", 1
    )
    accumulated_loss = 0
    opt.zero_grad()

    for batch_idx, batch in enumerate(train_loader):

        if cfg.data.use_pre_encoded:
            z_crown_omage = batch["z_crown_omage"].to(device)
        else:
            crown_omage = batch["crown_omage"].to(device)
            with torch.no_grad():
                z_crown_omage = vae.encode_image(crown_omage)

        condition_pc = batch["condition_pc"][:,:,:3].to(device)
        model_kwargs = dict(cond1=condition_pc)

        # Forward pass with AMP if enabled
        if scaler is not None:
            with autocast(device_type="cuda"):
                loss_dict = transport.training_losses(
                    model, z_crown_omage, model_kwargs
                )
                loss = loss_dict["loss"].mean() / gradient_accumulation_steps
        else:
            loss_dict = transport.training_losses(model, z_crown_omage, model_kwargs)
            loss = loss_dict["loss"].mean() / gradient_accumulation_steps

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accumulated_loss += loss.item()

        # Update weights after accumulating gradients
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()

            opt.zero_grad()
            update_ema(ema, model.module, decay=cfg.training.ema_decay)

            running_loss += accumulated_loss
            train_steps += 1
            accumulated_loss = 0

            # Validation and logging
            if train_steps % cfg.logging.log_every == 0:
                running_loss, start_time = log_and_validate(
                    train_steps,
                    running_loss,
                    start_time,
                    model,
                    ema,
                    vae,
                    transport,
                    transport_sampler,
                    val_loader,
                    cfg,
                    device,
                    scaler,
                    zs,
                    model_fn,
                    chamfer_loss,
                    n_chamfer_batches,
                    local_batch_size,
                    mesh_output_dir,
                    val_dataset,
                    logger,
                    epoch,
                )

            # Checkpointing
            if train_steps % cfg.logging.ckpt_every == 0 and train_steps > 0:
                checkpoint_model(
                    model, ema, opt, cfg, scaler, logger, checkpoint_dir, train_steps
                )

    return train_steps, running_loss, start_time


def log_and_validate(
    train_steps,
    running_loss,
    start_time,
    model,
    ema,
    vae,
    transport,
    transport_sampler,
    val_loader,
    cfg,
    device,
    scaler,
    zs,
    model_fn,
    chamfer_loss,
    n_chamfer_batches,
    local_batch_size,
    mesh_output_dir,
    val_dataset,
    logger,
    epoch,
):
    """Handle logging and validation"""
    # Calculate training speed
    torch.cuda.synchronize()
    steps_per_sec = cfg.logging.log_every / (time() - start_time)

    # Reduce loss across processes
    avg_loss = torch.tensor(running_loss / cfg.logging.log_every, device=device)
    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
    avg_loss = avg_loss.item() / dist.get_world_size()

    # Run validation
    avg_val_loss, avg_chamfer_loss = run_validation(
        model,
        vae,
        transport,
        transport_sampler,
        val_loader,
        cfg,
        device,
        scaler,
        train_steps,
        zs,
        model_fn,
        chamfer_loss,
        n_chamfer_batches,
        local_batch_size,
        mesh_output_dir if is_main_process() else None,
        val_dataset,
        logger,
        epoch,
    )

    # Log results
    logger.info(
        f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, "
        f"Val Loss: {avg_val_loss:.4f}, Chamfer Loss: {avg_chamfer_loss}, "
        f"Train Steps/Sec: {steps_per_sec:.2f}"
    )

    # Wandb logging
    if cfg.wandb.enabled and is_main_process():
        wandb_utils.log(
            {
                "train loss": avg_loss,
                "val loss": avg_val_loss,
                "train steps/sec": steps_per_sec,
            },
            step=train_steps,
        )

    # Reset and return
    model.train()
    return 0, time()  # Reset running_loss and start_time


def checkpoint_model(model, ema, opt, cfg, scaler, logger, checkpoint_dir, train_steps):
    """Save model checkpoint"""
    if is_main_process():
        checkpoint_path = save_sit_checkpoint(
            model, ema, opt, cfg, scaler, checkpoint_dir, train_steps
        )
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    dist.barrier()


def calculate_scaling_factor(
    train_loader, vae, device, n_batches=10, original_scaling_factor=1.0
):
    """
    Calculate scaling factor from first n batches of training data.

    Temporarily sets VAE scaling factor to 1.0 to get raw encoded variance,
    then compounds with the original scaling factor.

    Args:
        train_loader: Training data loader
        vae: VAE model
        device: Device to use
        n_batches: Number of batches to use (0 means skip calculation)
        original_scaling_factor: The original scaling factor to compound with

    Returns:
        scaling_factor: Calculated scaling factor or None if n_batches == 0
    """
    if n_batches == 0:
        return None

    print(f"Calculating scaling factor from first {n_batches} batches...")
    vae.eval()

    # Temporarily set scaling factor to 1.0 to get raw encoded data
    original_sf = vae.vae.config.scaling_factor
    vae.vae.config.scaling_factor = 1.0

    all_encoded = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= n_batches:
                break

            crown_omage = batch["crown_omage"].to(device)
            z_crown_omage = vae.encode_image(crown_omage)
            all_encoded.append(z_crown_omage.detach().cpu())

    # Concatenate all encoded samples
    all_encoded = torch.cat(all_encoded, dim=0)

    # Calculate scaling factor as the inverse of the std dev
    raw_std = torch.std(all_encoded).item()
    new_scaling_factor = 1.0 / raw_std

    # Compound with original scaling factor
    final_scaling_factor = new_scaling_factor * original_scaling_factor

    print(f"Raw std from data: {raw_std:.6f}")
    print(f"Calculated scaling factor: {new_scaling_factor:.6f}")
    print(f"Original scaling factor: {original_scaling_factor:.6f}")
    print(f"Final scaling factor (compounded): {final_scaling_factor:.6f}")

    # Restore original scaling factor
    vae.vae.config.scaling_factor = original_sf

    return final_scaling_factor


def main(cfg):
    """
    Trains a new SiT model using configuration.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup distributed training
    setup_distributed()
    device = get_device_id()
    rank = dist.get_rank()

    # Setup seed
    seed = cfg.training.global_seed * dist.get_world_size() + rank
    setup_seed(seed)

    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Calculate batch sizes
    assert (
        cfg.training.global_batch_size % dist.get_world_size() == 0
    ), f"Batch size must be divisible by world size."
    local_batch_size = int(cfg.training.global_batch_size // dist.get_world_size())
    gradient_accumulation_steps = getattr(
        cfg.training, "gradient_accumulation_steps", 1
    )

    # Initialize AMP scaler
    scaler = GradScaler() if cfg.training.get("use_amp", False) else None

    # Setup experiment directory and logging
    if is_main_process():
        experiment_dir, checkpoint_dir, mesh_output_dir, experiment_name = (
            create_experiment_dirs(cfg.logging.results_dir, cfg.model.name, "sit")
        )
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        save_config(cfg, experiment_dir)

        if cfg.wandb.enabled:
            wandb_experiment_name = cfg.wandb.get("experiment_name", None)
            final_experiment_name = (
                wandb_experiment_name if wandb_experiment_name else experiment_name
            )
            setup_wandb(cfg, final_experiment_name, cfg.wandb.project)
    else:
        logger = create_logger(None)
        checkpoint_dir = None
        mesh_output_dir = None

    # Create model
    assert (
        cfg.model.image_size % 8 == 0
    ), "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = cfg.model.image_size // 8
    model = SiT_models[cfg.model.name](
        input_size=latent_size,
        in_channels=cfg.model.in_channels,
    )

    # Create EMA and optimizer
    ema = deepcopy(model).to(device)
    ema.eval()  # EMA should always be in eval mode
    requires_grad(ema, False)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    # Load checkpoint if specified
    if cfg.resume.ckpt_path is not None:
        cfg, scaler = load_sit_checkpoint(
            cfg, model, ema, opt, cfg.resume.ckpt_path, device, scaler
        )

    model = DDP(model.to(device), device_ids=[device])

    # Create transport
    transport = create_transport(
        cfg.transport.path_type,
        cfg.transport.prediction,
        cfg.transport.loss_weight,
        cfg.transport.train_eps,
        cfg.transport.sample_eps,
    )
    transport_sampler = Sampler(transport)

    # Load VAE
    vae_cfg = OmegaConf.load(cfg.vae.config_path)

    # Determine scaling factor
    vae_scaling_factor = cfg.vae.scaling_factor
    if cfg.vae.n_batches_scaling > 0:
        # Will calculate after loading data
        calculate_scaling = True
    else:
        calculate_scaling = False

    vae = AutoEncoder(scaling_factor=vae_scaling_factor, **vae_cfg).to(device)
    load_ddp(vae, cfg.vae.ckpt_path)
    vae.eval()

    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup data
    if cfg.data.use_pre_encoded:
        train_dataset = PreEncodedConditionalGeneration(
            cfg.data.data_path,
            cfg.data.encoded_data_path,
            cfg.data.orig_data_path,
            split="train",
            debug=cfg.debug.enabled,
            conditions_in_memory=True,
        )
        val_dataset = PreEncodedConditionalGeneration(
            cfg.data.data_path,
            cfg.data.encoded_data_path,
            cfg.data.orig_data_path,
            split="val",
            debug=cfg.debug.enabled,
            conditions_in_memory=True,
        )
    else:
        train_dataset = Conditional_Generation(cfg.data.data_path, cfg.data.orig_data_path, split="train")
        val_dataset = Conditional_Generation(cfg.data.data_path, cfg.data.orig_data_path, split="val")

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=cfg.training.global_seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    # Calculate scaling factor if needed
    if calculate_scaling and not cfg.data.use_pre_encoded and is_main_process():
        new_scaling_factor = calculate_scaling_factor(
            train_loader, vae, device, cfg.vae.n_batches_scaling, cfg.vae.scaling_factor
        )
        vae.vae.config.scaling_factor = new_scaling_factor

    if cfg.data.use_pre_encoded:
        scaling_factor_path = cfg.vae.ckpt_path.replace(".pt", "_scaling_factor.txt")
        pre_calculated_scaling_factor = float(open(scaling_factor_path).read().strip())
        vae.vae.config.scaling_factor = pre_calculated_scaling_factor
    logger.info(f"Updated VAE scaling factor to {vae.vae.config.scaling_factor:.6f}")
    dist.barrier()

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=cfg.training.global_seed,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    logger.info(
        f"Dataset contains {len(train_dataset):,} images ({cfg.data.encoded_data_path if cfg.data.use_pre_encoded else cfg.data.data_path})"
    )

    # Prepare models for training
    update_ema(ema, model.module, decay=0)
    model.train()

    # Training variables
    train_steps, running_loss = 0, 0
    start_time = time()
    chamfer_loss = omages_utils.ChamferLoss()
    n_chamfer_batches = cfg.validation.chamfer_batches

    # Setup for sampling
    zs = torch.randn(local_batch_size, cfg.model.in_channels, latent_size, latent_size, device=device)
    use_cfg = cfg.cfg.scale > 1.0
    model_fn = ema.forward_with_cfg if use_cfg else ema.forward
    if use_cfg:
        zs = torch.cat([zs, zs], 0)

    if is_main_process():
        logger.info(f"Training for {cfg.training.epochs} epochs...")
        logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        logger.info(f"AMP enabled: {cfg.training.get('use_amp', False)}")
        logger.info(f"VAE scaling factor: {vae.vae.config.scaling_factor:.6f}")

    # Training loop
    for epoch in range(cfg.training.epochs):
        train_sampler.set_epoch(epoch)
        if is_main_process():
            logger.info(f"Beginning epoch {epoch}...")

        train_steps, running_loss, start_time = train_epoch(
            epoch,
            model,
            ema,
            opt,
            train_loader,
            vae,
            transport,
            transport_sampler,
            val_loader,
            cfg,
            device,
            scaler,
            logger,
            train_steps,
            running_loss,
            start_time,
            zs,
            model_fn,
            chamfer_loss,
            n_chamfer_batches,
            local_batch_size,
            mesh_output_dir,
            val_dataset,
            checkpoint_dir,
        )


def _do_visualization(
    out_samples,
    val_batch,
    local_batch_size,
    mesh_output_dir,
    epoch,
    train_steps,
    val_dataset,
    cfg,
    device,
    logger,
):
    """Handle all visualization logic (main process only)"""
    import matplotlib.pyplot as plt

    logger.info("Generating EMA samples done.")

    if cfg.wandb.enabled:
        wandb_utils.log_image(out_samples, train_steps)

    log_output = wandb_utils.array2grid(out_samples)
    viz_output_path = f"{mesh_output_dir}/{epoch:03d}"
    os.makedirs(viz_output_path, exist_ok=True)

    plt.imshow(log_output)
    plt.savefig(f"{mesh_output_dir}/sample_{train_steps:07d}.png")
    plt.close()
    del log_output

    out_samples_numpy = out_samples[:local_batch_size].detach().cpu().numpy()
    np.save(
        os.path.join(viz_output_path, f"out_samples_{train_steps:07d}.npy"),
        out_samples_numpy,
    )

    copy_orig_data_files(val_batch, viz_output_path, local_batch_size)


def run_validation(
    model,
    vae,
    transport,
    transport_sampler,
    val_loader,
    cfg,
    device,
    scaler,
    train_steps,
    zs,
    model_fn,
    chamfer_loss,
    n_chamfer_batches,
    local_batch_size,
    mesh_output_dir,
    val_dataset,
    logger,
    epoch,
):
    """Run validation loop"""
    # Add barrier at start to ensure all processes enter validation together
    dist.barrier()

    if is_main_process():
        logger.info("Starting validation...")
    model.eval()
    val_loss, val_steps, val_chamfer_loss, val_chamfer_steps = 0, 0, 0, 0

    # Limit validation batches to prevent memory issues
    max_val_batches = min(len(val_loader), 50)
    use_cfg = cfg.cfg.scale > 1.0

    for i, val_batch in enumerate(
        tqdm(val_loader, desc="Validation") if is_main_process() else val_loader
    ):
        if i >= max_val_batches:  # Break early to prevent hanging
            break

        crown_pc = val_batch["crown_pc"].to(device)
        condition_pc = val_batch["condition_pc"][:,:,:3].to(device)
        model_kwargs = dict(cond1=condition_pc)

        with torch.no_grad():
            if cfg.data.use_pre_encoded:
                z_crown_omage = val_batch["z_crown_omage"].to(device)
            else:
                crown_omage = val_batch["crown_omage"].to(device)
                z_crown_omage = vae.encode_image(crown_omage)
                del crown_omage

            # Validation loss calculation
            if scaler is not None:
                with autocast(device_type="cuda"):
                    loss_dict = transport.training_losses(
                        model, z_crown_omage, model_kwargs
                    )
            else:
                loss_dict = transport.training_losses(
                    model, z_crown_omage, model_kwargs
                )

        loss = loss_dict["loss"].mean()
        val_loss += loss.item()
        val_steps += 1
        
        del loss_dict, z_crown_omage, condition_pc
        
        # Chamfer loss calculation
        if train_steps % cfg.logging.chamfer_every == 0 and i < n_chamfer_batches:
            if use_cfg:
                model_kwargs["cfg_scale"] = cfg.cfg.scale
                model_kwargs["cond1"] = torch.cat(
                    [model_kwargs["cond1"], model_kwargs["cond1"]], dim=0
                )
            with torch.no_grad():
                if scaler is not None:
                    with autocast(device_type="cuda"):
                        sample_fn = transport_sampler.sample_ode()
                        samples = sample_fn(zs, model_fn, **model_kwargs)[-1]
                else:
                    sample_fn = transport_sampler.sample_ode()
                    samples = sample_fn(zs, model_fn, **model_kwargs)[-1]
                del sample_fn

                if use_cfg:
                    samples, _ = samples.chunk(2, dim=0)

                # Clear cache before collective operations
                torch.cuda.empty_cache()

                # Synchronize before collective operations
                dist.barrier()

                # Decode samples
                samples = vae.decode_latent(samples)

                # Gather samples across all processes - ensure consistent tensor sizes
                world_size = dist.get_world_size()
                out_samples = torch.zeros(
                    (
                        samples.shape[0] * world_size,
                        3,
                        cfg.model.image_size,
                        cfg.model.image_size,
                    ),
                    device=device,
                    dtype=samples.dtype,  # Ensure consistent dtype
                )

                try:
                    dist.all_gather_into_tensor(out_samples, samples)
                except Exception as e:
                    if is_main_process():
                        logger.error(f"all_gather_into_tensor failed: {e}")
                    del samples, out_samples
                    continue

                # Gather crown_pc across all processes
                crown_pc_gathered = torch.zeros(
                    (
                        crown_pc.shape[0] * world_size,
                        crown_pc.shape[1],
                        crown_pc.shape[2],
                    ),
                    device=device,
                    dtype=crown_pc.dtype,
                )

                try:
                    dist.all_gather_into_tensor(crown_pc_gathered, crown_pc)
                except Exception as e:
                    if is_main_process():
                        logger.error(f"crown_pc all_gather_into_tensor failed: {e}")
                    del samples, out_samples, crown_pc_gathered, model_kwargs
                    continue

                # Subsample to point clouds
                out_samples_pc = []
                for b in range(out_samples.shape[0]):
                    vert = out_samples[b].transpose(0, 2).reshape(-1, 3).detach().cpu().numpy()
                    n_sample = min(10000, vert.shape[0])
                    if vert.shape[0] > n_sample:
                        vert_idx = fpsample.bucket_fps_kdline_sampling(vert, n_sample, h=7)
                        vert = vert[vert_idx]
                    out_samples_pc.append(vert)
                out_samples_pc = torch.Tensor(np.stack(out_samples_pc)).to(device)

                chamfer_normalized = chamfer_validation(
                    out_samples_pc, crown_pc_gathered, chamfer_loss, device
                ).item()
                
                val_chamfer_loss += (chamfer_normalized ** 0.5) * val_dataset.GLOBAL_AVERAGE_VERTEX_NORM * 100
                val_chamfer_steps += 1

                del out_samples, out_samples_pc, crown_pc_gathered, samples
                torch.cuda.empty_cache()

        # Sample generation and visualization - only when needed
        if train_steps % cfg.logging.sample_every == 0 and i == 0:
            with torch.no_grad():
                if scaler is not None:
                    with autocast(device_type="cuda"):
                        sample_fn = transport_sampler.sample_ode()
                        samples_trajectory = sample_fn(zs, model_fn, **model_kwargs)
                else:
                    sample_fn = transport_sampler.sample_ode()
                    samples_trajectory = sample_fn(zs, model_fn, **model_kwargs)
                del sample_fn

                # Use final sample for visualization
                samples = samples_trajectory[-1]
                del samples_trajectory
                
                samples = vae.decode_latent(samples)

                # All processes participate in gather
                world_size = dist.get_world_size()
                out_samples = torch.zeros(
                    (
                        samples.shape[0] * world_size,
                        3,
                        cfg.model.image_size,
                        cfg.model.image_size,
                    ),
                    device=device,
                    dtype=samples.dtype,
                )
                dist.all_gather_into_tensor(out_samples, samples)
                del samples, model_kwargs

            if is_main_process():
                _do_visualization(
                    out_samples,
                    val_batch,
                    local_batch_size,
                    mesh_output_dir,
                    epoch,
                    train_steps,
                    val_dataset,
                    cfg,
                    device,
                    logger,
                )

            del out_samples
            torch.cuda.empty_cache()
        
        del crown_pc, val_batch
        
        if i % 10 == 0:
            torch.cuda.empty_cache()

    dist.barrier()

    avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
    avg_chamfer_loss = (
        val_chamfer_loss / val_chamfer_steps
        if val_chamfer_steps > 0
        else 0
    )

    return avg_val_loss, avg_chamfer_loss


def save_trajectory_samples(
    samples_trajectory,
    vae,
    output_dir,
    train_steps,
    global_average_vertex_norm,
    logger,
):
    """
    Save mesh samples from trajectory.

    Args:
        samples_trajectory: List of latent samples at each timestep
        vae: VAE model for decoding
        output_dir: Directory to save samples
        train_steps: Current training step for naming
        device: Device to use
        logger: Logger instance
    """
    import trimesh

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, z_t in enumerate(samples_trajectory):
            # Decode latent to omage (3-channel: position or RGB)
            decoded_sample = vae.decode_latent(z_t[:1])  # Shape: (B, 3, H, W)

            # Convert to numpy
            decoded_numpy = decoded_sample.detach().cpu().numpy()  # (B, 3, H, W)

            # Process each sample in the batch
            for batch_idx in range(decoded_numpy.shape[0]):
                sample_data = decoded_numpy[batch_idx]  # (3, H, W)
                sample_data = np.transpose(sample_data, (1, 2, 0))  # (H, W, 3)

                # Convert to mesh using omages_utils
                meshing_result = omages_utils.meshing_objamage(sample_data)

                # Create trimesh from vertices and faces
                mesh = trimesh.Trimesh(
                    vertices=meshing_result["vert"] * global_average_vertex_norm,
                    faces=meshing_result["face"],
                )

                # Save as PLY
                filename = f"{output_dir}/trajectory_step_{idx:03d}_batch_{batch_idx:03d}_step_{train_steps:07d}.ply"
                mesh.export(filename)

    logger.info(f"Saved trajectory samples to {output_dir}")


if __name__ == "__main__":
    # Setup environment
    setup_environment()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/SiT_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument("--data-path", type=str, help="Override data path")
    parser.add_argument("--orig-data-path", type=str, help="Override original (mesh) data root")
    parser.add_argument("--vae-ckpt", type=str, help="Override VAE checkpoint path")
    parser.add_argument("--results-dir", type=str, help="Override results directory")
    parser.add_argument("--wandb", action="store_true", default=None, help="Enable wandb logging")
    parser.add_argument("--debug", action="store_true", default=None, help="Enable debug mode")
    parser.add_argument("--ckpt", type=str, help="Resume from checkpoint")

    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config, args)

    # Validation
    if cfg.data.use_pre_encoded and cfg.data.encoded_data_path is None:
        raise ValueError(
            "encoded_data_path must be specified when use_pre_encoded is true"
        )

    if not cfg.data.data_path:
        raise ValueError("data_path must be specified")

    if not cfg.vae.ckpt_path:
        raise ValueError("vae.ckpt_path must be specified")

    print("logging every", cfg.logging.log_every, "steps")

    main(cfg)
    cleanup_distributed()
