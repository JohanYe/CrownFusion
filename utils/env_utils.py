"""Environment setup utilities"""

import os


def setup_environment():
    """Setup common environment variables"""
    # Distributed training defaults
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
