"""
Adversarial Attacks Implementation.

This module implements gradient-based adversarial attacks for generating adversarial examples.
Supported attacks:
1. Fast Gradient Sign Method (FGSM): A single-step attack.
2. Projected Gradient Descent (PGD): An iterative, stronger variant of FGSM.
"""
import torch
import torch.nn as nn

def fgsm_attack(model, data, target, epsilon):
    """
    Fast Gradient Sign Method (FGSM) attack.
    
    Generates adversarial examples by perturbing the input data in the direction of the 
    gradient of the loss function with respect to the input.
    
    Formula:
        x_adv = x + epsilon * sign(nabla_x J(theta, x, y))
        
    Args:
        model (nn.Module): The neural network model under attack.
        data (torch.Tensor): Input data tensor (Batch, Features) or (Batch, C, H, W).
        target (torch.Tensor): True class labels.
        epsilon (float): Perturbation magnitude (L-inf norm constraint).
        
    Returns:
        torch.Tensor: Adversarial examples.
    """
    # Create a copy of data to avoid modifying the original
    data = data.clone().detach()
    data.requires_grad = True

    # Forward pass
    output = model(data)
    loss = nn.CrossEntropyLoss()(output, target)

    # Zero all existing gradients
    model.zero_grad()

    # Backward pass
    loss.backward()

    # Collect data gradient
    data_grad = data.grad.data

    # Create the perturbed image
    # x_adv = x + epsilon * sign(grad)
    sign_data_grad = data_grad.sign()
    perturbed_data = data + epsilon * sign_data_grad

    return perturbed_data

def pgd_attack(model, data, target, epsilon, alpha=0.01, num_iter=10):
    """
    Projected Gradient Descent (PGD) attack.
    
    An iterative version of FGSM. In each step, it applies a small perturbation (alpha)
    and projects the result back into the epsilon-ball around the original input.
    
    Algorithm:
        x_0 = x
        x_{t+1} = Clip_{x, epsilon} (x_t + alpha * sign(nabla_x J(theta, x_t, y)))
        
    Args:
        model (nn.Module): The neural network model.
        data (torch.Tensor): Input data tensor.
        target (torch.Tensor): True class labels.
        epsilon (float): Maximum allowed perturbation (L-inf norm).
        alpha (float): Step size per iteration.
        num_iter (int): Number of attack iterations.
        
    Returns:
        torch.Tensor: Stronger adversarial examples.
    """
    original_data = data.clone().detach()
    perturbed_data = data.clone().detach()

    for _ in range(num_iter):
        perturbed_data.requires_grad = True

        output = model(perturbed_data)
        loss = nn.CrossEntropyLoss()(output, target)

        model.zero_grad()
        loss.backward()

        data_grad = perturbed_data.grad.data
        
        # Step
        perturbed_data = perturbed_data + alpha * data_grad.sign()

        # Projection (Clip to epsilon ball around original data)
        perturbation = torch.clamp(perturbed_data - original_data, min=-epsilon, max=epsilon)
        perturbed_data = original_data + perturbation

        # Detach to prevent graph buildup
        perturbed_data = perturbed_data.detach()

    return perturbed_data
