import os
import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize distributed training"""
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("NCCL_TIMEOUT", "12000")  # Increased timeout for NCCL

    dist.init_process_group("nccl")


def get_device_id():
    """Get the appropriate device for current rank"""
    if torch.cuda.is_available():
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if current process is the main process (rank 0)"""
    return not dist.is_initialized() or dist.get_rank() == 0
