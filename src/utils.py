"""
Utility Functions.

This module provides helper functions for:
1. Reproducibility (seeding).
2. Hardware configuration (device selection).
"""
import random
import numpy as np
import torch

def set_seed(seed):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    
    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """
    Detects and returns the available computation device.
    
    Returns:
        torch.device: 'cuda' if GPU is available, otherwise 'cpu'.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device
