import os
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

def set_seed(seed: int = 42) -> None:
    """
    Sets the seed for reproducibility across numpy, random, and torch.
    
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")

class TensorboardLogger:
    """
    A simple wrapper for Tensorboard logging.
    """
    def __init__(self, log_dir: str = "logs/"):
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Logs a scalar value to Tensorboard."""
        self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        """Closes the SummaryWriter."""
        self.writer.close()
