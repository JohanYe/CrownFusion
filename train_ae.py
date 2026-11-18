import os
import argparse
import datetime
import random
from glob import glob
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
from torchvision import transforms
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt

os.environ["OMP_NUM_THREADS"] = "1"

# Import shared utilities
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
from utils.checkpoint_utils import save_checkpoint
from utils.training_utils import log_losses

# Import project modules
import omages_utils
import wandb
import sit_utils.wandb_utils as wandb_utils
from dataset import Only_Omages
from sit_utils.models import AutoEncoder
import fpsample
import igl
import hydra

#################################################################################
#                         AutoEncoder Specific Utilities                        #
#################################################################################


def configure_optimizers(model, cfg, is_distributed=False):
    """Configure optimizer and scheduler for AutoEncoder"""
    param_groups = []
    trainable_params = (
        model.module.trainable_params if is_distributed else model.trainable_params
    )
    for params, lr_scale in trainable_params:
        lr = model.module.lr * lr_scale if is_distributed else model.lr * lr_scale
        param_groups.append({"params": params, "lr": lr})

    optimizer = hydra.utils.instantiate(cfg.model.optimizer, param_groups)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.99, last_epoch=-1
    )
    return optimizer, scheduler


def compute_chamfer_loss_ae(pred, gt_samples, chamfer_loss, device, occupancy=None):
    """Compute chamfer loss for autoencoder validation"""
    recon_pc = []
    for i in range(pred.size(0)):
        vert = pred[i].transpose(0, 2).reshape(-1, 3).detach().cpu().numpy()
        
        # Filter out unoccupied (zero) points if occupancy mask is provided
        if occupancy is not None:
            occ = occupancy[i].reshape(-1).cpu().numpy() > 0.5
            vert = vert[occ]
        else:
            # Remove points that are all zeros
            vert = vert[np.abs(vert).sum(axis=1) > 1e-6]
        
        # Sample 4096 points via FPS
        if len(vert) >= 4096:
            vert_idx = fpsample.bucket_fps_kdline_sampling(vert, 4096, h=7)
            vert = vert[vert_idx]
        
        recon_pc.append(vert)
    recon_pc = torch.Tensor(np.stack(recon_pc)).to(device)
    return chamfer_loss(gt_samples, recon_pc)


def save_mesh_samples(sample_output, sample_occupancy, epoch_mesh_output_dir, num_samples):
    """Save mesh samples for visualization"""
    os.makedirs(epoch_mesh_output_dir, exist_ok=True)

    print(f"Saving mesh samples... {num_samples} samples to {epoch_mesh_output_dir}")
    for i in range(num_samples):
        sample_mesh = omages_utils.meshing_objamage(
            {
                "position": np.transpose(
                    sample_output[i].float().cpu().numpy(), (1, 2, 0)
                ),
                "occupancy": np.transpose(
                    sample_occupancy[i].float().cpu().numpy(), (1, 2, 0)
                ),
            }
        )
        igl.write_triangle_mesh(
            f"{epoch_mesh_output_dir}/sample_mesh_{i}.obj",
            sample_mesh["vert"],
            sample_mesh["face"],
        )


#################################################################################
#                              Training Classes                                 #
#################################################################################


class AutoEncoderTrainer:
    """AutoEncoder trainer class"""

    def __init__(self, cfg, model, train_loader, val_loader, device):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Setup distributed training
        self.is_distributed = torch.cuda.device_count() > 1
        if self.is_distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[device]
            )

        # Setup optimizer and scheduler
        self.optimizer, self.scheduler = configure_optimizers(
            self.model, cfg, self.is_distributed
        )

        # Setup AMP
        self.use_amp = cfg.trainer.get("use_amp", False) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

        # Setup gradient accumulation
        self.accumulation_steps = cfg.trainer.get("gradient_accumulation_steps", 1)

        # Training state
        self.start_epoch = 0
        self.rank = dist.get_rank() if self.is_distributed else 0

        print(f"Using AMP: {self.use_amp}")
        if self.rank == 0 and self.use_amp:
            print("Using Automatic Mixed Precision (AMP)")

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss_dict = {"loss": 0.0, "kl_loss": 0.0, "mse_loss": 0.0}

        # Log current learning rate
        current_lr = self.optimizer.param_groups[0]["lr"]
        if self.rank == 0:
            print(f"Epoch {epoch+1}, Current Learning Rate: {current_lr:.2e}")

        for i, batch in enumerate(self.train_loader):
            loss = self._train_step(batch, i)

            # Log losses
            log_losses(
                {"loss": loss, "mse_loss": loss},  # For AE, loss is primarily MSE
                batch["omage"].shape[0],
                running_loss_dict,
            )

            if self.cfg.debug_mode and i > 20:
                break

        # Calculate averages
        avg_loss = running_loss_dict["loss"] / len(self.train_loader.dataset)
        avg_mse_loss = running_loss_dict["mse_loss"] / len(self.train_loader.dataset)

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.rank == 0:
            print(
                f"{current_time}: Epoch {epoch+1}, Training Loss: {avg_loss:.4f}, MSE Loss: {avg_mse_loss:.4f}"
            )

        return avg_loss, avg_mse_loss

    def _train_step(self, batch, batch_idx):
        """Single training step"""
        x = batch["omage"].to(self.device)
        batch_size = x.shape[0]

        # Zero gradients at start of accumulation cycle
        if batch_idx % self.accumulation_steps == 0:
            self.optimizer.zero_grad()

        # Forward pass with AMP
        with autocast(device_type="cuda", enabled=self.use_amp):
            posterior = (
                self.model.module.vae.encode(x).latent_dist
                if self.is_distributed
                else self.model.vae.encode(x).latent_dist
            )
            z = posterior.sample()
            pred = (
                self.model.module.vae.decode(z).sample
                if self.is_distributed
                else self.model.vae.decode(z).sample
            )
            # Compute MSE loss over entire image (including corners)
            mse_loss = (
                torch.nn.functional.mse_loss(pred, x, reduction="sum")
                / batch_size
            )

            # Scale loss by accumulation steps for proper averaging
            loss = mse_loss / self.accumulation_steps

        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights at end of accumulation cycle
        if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(
            self.train_loader
        ):
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

        return loss * self.accumulation_steps  # Return unscaled loss for logging

    def validate(self, epoch):
        """Run validation"""
        self.model.eval()
        running_val_loss_dict = {"loss": 0.0, "kl_loss": 0.0, "mse_loss": 0.0}
        chamfer_loss = omages_utils.ChamferLoss()
        chamfer_loss_total, total_val_samples = 0.0, 0

        sample_input, sample_output, sample_occupancy = None, None, None

        for j, batch in enumerate(self.val_loader):
            val_loss, pred = self._validation_step(batch)

            # Store samples for visualization
            if j == 0:
                sample_input = batch["omage"][:8]
                sample_output = pred[:8]
                sample_occupancy = batch["occupancy"][:8]

            # Compute chamfer loss
            chamfer_val = self._compute_validation_chamfer(batch, pred, chamfer_loss)
            if chamfer_val is not None:
                chamfer_loss_total += chamfer_val
                total_val_samples += batch["omage"].size(0)

            # Log losses
            log_losses(
                {"loss": val_loss, "mse_loss": val_loss},
                batch["omage"].shape[0],
                running_val_loss_dict,
            )

            if self.cfg.debug_mode and j > 20:
                break

        # Calculate averages
        avg_val_loss = running_val_loss_dict["loss"] / len(self.val_loader.dataset)
        avg_chamfer_loss = (
            chamfer_loss_total / total_val_samples if total_val_samples > 0 else 0
        )

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.rank == 0:
            print(
                f"{current_time}: Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}, Chamfer Loss: {avg_chamfer_loss:.4f}"
            )

        return avg_val_loss, avg_chamfer_loss, sample_input, sample_output, sample_occupancy

    def _validation_step(self, batch):
        """Single validation step"""
        x = batch["omage"].to(self.device)
        batch_size = x.shape[0]

        with torch.no_grad():
            with autocast(device_type="cuda", enabled=self.use_amp):
                posterior = (
                    self.model.module.vae.encode(x).latent_dist
                    if self.is_distributed
                    else self.model.vae.encode(x).latent_dist
                )
                z = posterior.sample()
                pred = (
                    self.model.module.vae.decode(z).sample
                    if self.is_distributed
                    else self.model.vae.decode(z).sample
                )

            # Compute MSE loss over entire image (including corners)
            mse_loss = (
                torch.nn.functional.mse_loss(pred, x, reduction="sum")
                / batch_size
            )

        return mse_loss, pred

    def _compute_validation_chamfer(self, batch, pred, chamfer_loss):
        """Compute chamfer loss for validation batch"""
        # Use pre-computed ground truth point clouds from batch
        if "gt_pc" not in batch:
            if self.rank == 0:
                print(f"WARNING: 'gt_pc' not found in batch. Available keys: {batch.keys()}")
            return None
        
        gt_pc = batch["gt_pc"].to(self.device)  # [batch, 2048, 3]
        return compute_chamfer_loss_ae(pred, gt_pc, chamfer_loss, self.device, occupancy=batch["occupancy"]).item()

    def step_scheduler(self):
        """Step the learning rate scheduler"""
        self.scheduler.step()

    def save_checkpoint(self, epoch, checkpoint_dir):
        """Save training checkpoint"""
        if self.rank == 0:
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            }
            if self.scaler is not None:
                checkpoint_data["scaler_state_dict"] = self.scaler.state_dict()

            checkpoint_path = save_checkpoint(
                checkpoint_data, checkpoint_dir, f"epoch_{epoch+1}"
            )
            print(f"Checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            if self.scaler is not None and "scaler_state_dict" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

            self.start_epoch = checkpoint["epoch"] + 1
            self.scheduler.last_epoch = self.start_epoch - 1

            print(
                f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {self.start_epoch}"
            )


#################################################################################
#                                Main Function                                  #
#################################################################################


def main():
    # Setup environment
    setup_environment()

    # Setup distributed training
    setup_distributed()
    device = get_device_id()
    rank = dist.get_rank()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="./configs/ae_config.yaml", help="Path to AE configuration file"
    )
    args = parser.parse_args()

    # Load configuration
    cfg = OmegaConf.load(args.config)

    # Setup seed
    setup_seed(cfg.seed)

    # Setup experiment directory and logging
    experiment_dir, checkpoint_dir, mesh_output_dir, experiment_name = (
        None,
        None,
        None,
        None,
    )

    if is_main_process():
        experiment_dir, checkpoint_dir, mesh_output_dir, experiment_name = (
            create_experiment_dirs(
                cfg.results_dir, cfg.model.network.model_id, "ae", cfg.debug_mode
            )
        )

        # Ensure checkpoint_dir exists even in debug mode
        if checkpoint_dir is None and cfg.debug_mode:
            # Create numbered experiment folder in debug directory
            os.makedirs("./debug", exist_ok=True)
            existing_experiments = glob("./debug/[0-9][0-9][0-9]-*")
            if existing_experiments:
                indices = [
                    int(os.path.basename(exp).split("-")[0]) for exp in existing_experiments
                ]
                experiment_index = max(indices) + 1
            else:
                experiment_index = 0
            
            model_string_name = cfg.model.network.model_id.replace("/", "-")
            random_suffix = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6))
            experiment_name = f"{experiment_index:03d}-{model_string_name}-ae-{random_suffix}"
            
            experiment_dir = f"./debug/{experiment_name}"
            checkpoint_dir = f"{experiment_dir}/checkpoints"
            mesh_output_dir = f"{experiment_dir}/mesh_samples"
            
            os.makedirs(experiment_dir, exist_ok=True)
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(mesh_output_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Experiment: {experiment_name}")
        print(f"Experiment Directory: {experiment_dir}")
        print(f"Checkpoint Directory: {checkpoint_dir}")
        print(f"Mesh Output Directory: {mesh_output_dir}")
        print(f"{'='*80}\n")

        create_logger(experiment_dir)

        if not cfg.debug_mode:
            # Save config
            save_config(cfg, experiment_dir, "ae_config.yaml")

            # Setup wandb
            setup_wandb(cfg, experiment_name, "omages_aekl")
    else:
        checkpoint_dir = None
        mesh_output_dir = None
        # Disable wandb for non-main processes
        os.environ["WANDB_MODE"] = "disabled"
        wandb.init(mode="disabled")

    # Setup data transforms
    transform = transforms.Compose(
        [
            # Add your transforms here if needed
        ]
    )

    # Create datasets
    train_dataset = Only_Omages(
        cfg.data.data_path, cfg.data.orig_data_path, data=cfg.data.omage_type, split="train", transform=transform
    )
    val_dataset = Only_Omages(cfg.data.data_path, cfg.data.orig_data_path, data=cfg.data.omage_type, split="val")

    # Create data loaders
    world_size = dist.get_world_size()
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.seed
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=cfg.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=0 if cfg.debug_mode else 4,
        sampler=train_sampler,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=0 if cfg.debug_mode else 4,
        sampler=val_sampler,
        drop_last=True,
    )

    # Create model
    model = AutoEncoder(scaling_factor=0.18215, **cfg).to(device)

    # Create trainer
    trainer = AutoEncoderTrainer(cfg, model, train_loader, val_loader, device)

    # Load checkpoint if specified
    if cfg.trainer.resume_from != "":
        trainer.load_checkpoint(cfg.trainer.resume_from)

    # Training loop
    for epoch in range(trainer.start_epoch, cfg.trainer.max_epochs):
        # Training
        avg_loss, avg_mse_loss = trainer.train_epoch(epoch)

        # Validation
        avg_val_loss, avg_chamfer_loss, sample_input, sample_output, sample_occupancy = trainer.validate(
            epoch
        )

        # Logging and visualization
        if is_main_process() and not cfg.debug_mode:
            # Log to wandb
            current_lr = trainer.optimizer.param_groups[0]["lr"]

            # Create visualization
            if sample_input is not None and sample_output is not None:
                log_input = wandb_utils.array2grid(sample_input)
                log_output = wandb_utils.array2grid(sample_output)

                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(log_input)
                axes[0].set_title("Input")
                axes[1].imshow(log_output)
                axes[1].set_title("Output")
                plt.tight_layout()

                wandb.log(
                    {
                        "Training Loss": avg_loss,
                        "Training MSE Loss": avg_mse_loss,
                        "Validation Loss": avg_val_loss,
                        "Chamfer Loss": avg_chamfer_loss,
                        "Learning Rate": current_lr,
                        "input_output_comparison": fig,
                        "epoch": epoch,
                    }
                )
                plt.close(fig)

        # Save checkpoint
        if (epoch + 1) % cfg.trainer.save_checkpoint_every_n_epoch == 0:
            if is_main_process() and checkpoint_dir is not None:
                trainer.save_checkpoint(epoch, checkpoint_dir)
                print(f"✓ Checkpoint saved at epoch {epoch + 1}")

        # Save mesh samples
        if (epoch + 1) % cfg.trainer.viz_every_n_epoch == 0 and is_main_process():
            if sample_output is not None and sample_occupancy is not None and mesh_output_dir is not None:
                epoch_mesh_output_dir = f"{mesh_output_dir}/{epoch:03d}"
                save_mesh_samples(
                    sample_output, sample_occupancy, epoch_mesh_output_dir, min(8, sample_output.size(0))
                )

        # Step scheduler
        trainer.step_scheduler()

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
