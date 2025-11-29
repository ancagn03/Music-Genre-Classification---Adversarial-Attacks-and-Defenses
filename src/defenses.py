"""
Defense Mechanisms against Adversarial Attacks.

This module implements inference-time defenses to improve model robustness.
Currently supported:
1. Feature Squeezing (Bit Depth Reduction): Reduces the precision of input features to destroy small adversarial perturbations.
"""
import torch

class FeatureSqueezing:
    """
    Feature Squeezing Defense (Bit Depth Reduction).
    
    This defense works by reducing the bit depth of the input data (quantization).
    The intuition is that adversarial perturbations are often small and high-frequency.
    By "squeezing" the feature space, we map many slightly different inputs to the same 
    quantized value, effectively filtering out the noise.
    """
    def __init__(self, bit_depth=4):
        """
        Args:
            bit_depth (int): The target bit depth for quantization (e.g., 4, 5, 8).
                             Lower values mean stronger squeezing but more loss of information.
        """
        self.bit_depth = bit_depth
        self.max_val = 2 ** bit_depth - 1

    def __call__(self, x):
        """
        Apply feature squeezing to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor (normalized [0, 1] or similar).
            
        Returns:
            torch.Tensor: Quantized (squeezed) tensor.
        """
        # Simple quantization logic:
        # x_squeezed = round(x * scale) / scale
        scale = self.max_val
        return torch.round(x * scale) / scale


