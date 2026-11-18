import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

# Import necessary modules from the main training script
from dataset import Only_Omages
from sit_utils.models import AutoEncoder
from utils.training_utils import load_ddp


def calculate_vae_scaling_factor(
    vae, data_path, orig_data_path, device="cuda", batch_size=8, num_workers=4, n_batches=10, checkpoint_path=None
):
    """
    Calculate VAE scaling factor to ensure latent distribution has std 1.

    Args:
        vae: The VAE model
        data_path: Path to the dataset
        orig_data_path: Root of the original (mesh) dataset
        device: Device to use for encoding
        batch_size: Batch size for processing
        n_batches: Number of batches to use for scaling factor calculation
        num_workers: Number of workers for data loading

    Returns:
        scaling_factor: Calculated scaling factor
    """
    print(f"\nCalculating VAE scaling factor using {n_batches} batches...")

    dataset = Only_Omages(data_path, orig_data_path, data="crown", split="train")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    all_latents = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="Collecting latents for scaling")
        ):
            if batch_idx >= n_batches:
                break

            crown_omage = batch["omage"].to(device)
            z_omage = vae.encode_image(crown_omage)
            all_latents.append(z_omage.cpu())
            torch.cuda.empty_cache()

    # Concatenate all latents
    all_latents = torch.cat(all_latents, dim=0)

    # Calculate standard deviation
    latent_std = all_latents.std()
    scaling_factor = 1.0 / latent_std

    print(f"Latent std: {latent_std:.6f}")
    print(f"Calculated scaling factor: {scaling_factor:.6f}")

    if checkpoint_path:
        scaling_factor_path = checkpoint_path.replace('.pt', '_scaling_factor.txt')
        with open(scaling_factor_path, 'w') as f:
            f.write(str(float(scaling_factor)))
        print(f"Scaling factor saved to {scaling_factor_path}")

    return float(scaling_factor)


def preprocess_and_save_encoded_crown_omages(
    data_path,
    orig_data_path,
    crown_vae_ckpt_path,
    device="cuda",
    batch_size=8,
    num_workers=4,
    calculate_scaling_factor=False,
    n_batches=10,
    ae_config_path="configs/ae_config.yaml",
):
    """
    Preprocess all crown omages by encoding them to latent space and saving to disk.

    Args:
        data_path: Path to the dataset
        orig_data_path: Root of the original (mesh) dataset
        crown_vae_ckpt_path: Path to the crown VAE checkpoint
        device: Device to use for encoding
        batch_size: Batch size for processing
        num_workers: Number of workers for data loading
        calculate_scaling_factor: Whether to calculate scaling factor from data
        n_batches: Number of batches to use for scaling factor calculation
    """
    print(f"Starting preprocessing of crown omages...")
    print(f"Data path: {data_path}")
    print(f"Crown VAE checkpoint: {crown_vae_ckpt_path}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")

    # Create output directory with "encoded_" prefix
    output_dir = os.path.join(
        os.path.dirname(data_path),
        f"encoded_{os.path.basename(data_path.rstrip('/'))}",
    )
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    print("Output directory created")

    # Process and save encoded crown omages for each split
    process_and_encode_crown_splits(
        data_path=data_path,
        orig_data_path=orig_data_path,
        vae_ckpt_path=crown_vae_ckpt_path,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        calculate_scaling_factor=calculate_scaling_factor,
        n_batches=n_batches,
        ae_config_path=ae_config_path,
    )


def process_and_encode_crown_splits(
    data_path,
    orig_data_path,
    vae_ckpt_path,
    output_dir,
    device="cuda",
    batch_size=8,
    num_workers=4,
    calculate_scaling_factor=False,
    n_batches=10,
    ae_config_path="configs/ae_config.yaml",
):
    # Create crown-specific directories
    os.makedirs(os.path.join(output_dir, "crown_train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "crown_val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "crown_test"), exist_ok=True)

    # Load crown VAE model
    print("Loading crown VAE model...")
    vae_cfg = OmegaConf.load(ae_config_path)

    # Use scaling factor of 1.0 for loading and potential scaling factor calculation
    vae = AutoEncoder(scaling_factor=1.0, **vae_cfg).to(device)
    load_ddp(vae, vae_ckpt_path)
    vae.eval()
    print("Crown VAE model loaded successfully")

    # Calculate scaling factor if requested
    if calculate_scaling_factor:
        scaling_factor = calculate_vae_scaling_factor(
            vae=vae,
            data_path=data_path,
            orig_data_path=orig_data_path,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            n_batches=n_batches,
            checkpoint_path=vae_ckpt_path,
        )
        # Update VAE with new scaling factor
        vae.vae.config.scaling_factor = scaling_factor
        print(f"Updated VAE scaling factor to {scaling_factor:.6f}")
    else:
        # Use default scaling factor if not calculating
        vae.vae.config.scaling_factor = 0.18215
        print("Using default VAE scaling factor: 0.18215")

    # Process each split
    for split in ["train", "val", "test"]:
        process_crown_split(
            split=split,
            data_path=data_path,
            orig_data_path=orig_data_path,
            output_dir=output_dir,
            vae=vae,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )


def process_crown_split(
    split, data_path, orig_data_path, output_dir, vae, device, batch_size, num_workers
):
    print(f"\nProcessing crown {split} split...")

    # Create dataset
    dataset = Only_Omages(data_path, orig_data_path, data="crown", split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Found {len(dataset)} samples in {split} split")

    # Process batches
    split_processed = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc=f"Encoding crown {split} batches")
        ):
            crown_omage = batch["omage"].to(device)
            sample_paths = batch["sample_id"]

            # Encode the crown omages
            z_omage = vae.encode_image(crown_omage)

            # Save each encoded omage individually
            for i in range(z_omage.shape[0]):
                sample_npy_file_name = sample_paths[i] + ".npy"
                encoded_omage = z_omage[i].cpu().numpy()

                # Save with crown prefix
                output_path = os.path.join(
                    output_dir, f"crown_{split}", sample_npy_file_name
                )
                np.save(output_path, encoded_omage)
                split_processed += 1

            # Clear GPU memory
            torch.cuda.empty_cache()

    print(f"Processed {split_processed} crown samples in {split} split")
    return split_processed


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess omages by encoding them to latent space"
    )
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "--orig-data-path",
        type=str,
        required=True,
        help="Root of the original dataset (<split>/<id>/ meshes)",
    )
    parser.add_argument(
        "--crown-vae-ckpt",
        type=str,
        help="Path to the crown VAE checkpoint (required for crown encoding)",
    )
    parser.add_argument(
        "--ae-config",
        type=str,
        default="configs/ae_config.yaml",
        help="AE architecture config used to build the VAE",
    )
    parser.add_argument(
        "--encode-crown",
        action="store_true",
        help="Encode crown omages using VAE",
    )
    parser.add_argument(
        "--calculate-scaling-factor",
        action="store_true",
        help="Calculate VAE scaling factor from data to ensure latent std=1",
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=10,
        help="Number of batches to use for scaling factor calculation",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for processing"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for encoding"
    )

    args = parser.parse_args()

    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = "cpu"

    # Check if input paths exist
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path does not exist: {args.data_path}")

    # Check if at least one encoding type is specified
    if not args.encode_crown:
        raise ValueError("Must specify --encode-crown")

    # Run crown encoding if requested
    if args.encode_crown:
        if not args.crown_vae_ckpt:
            raise ValueError("--crown-vae-ckpt is required when using --encode-crown")

        if not os.path.exists(args.crown_vae_ckpt):
            raise FileNotFoundError(
                f"Crown VAE checkpoint does not exist: {args.crown_vae_ckpt}"
            )

        preprocess_and_save_encoded_crown_omages(
            data_path=args.data_path,
            orig_data_path=args.orig_data_path,
            crown_vae_ckpt_path=args.crown_vae_ckpt,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            calculate_scaling_factor=args.calculate_scaling_factor,
            n_batches=args.n_batches,
            ae_config_path=args.ae_config,
        )


if __name__ == "__main__":
    main()
