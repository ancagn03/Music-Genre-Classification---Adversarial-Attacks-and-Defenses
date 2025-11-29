"""
Adversarial attacks implementation.
Includes FGSM and PGD (Iterative FGSM) attacks.
"""
import torch
import torch.nn as nn

def fgsm_attack(model, data, target, epsilon):
    """
    Fast Gradient Sign Method (FGSM) attack.
    Perturbs the data by epsilon in the direction of the gradient sign.
    
    Args:
        model: The neural network model.
        data: Input tensor (Batch, Features) or (Batch, C, H, W).
        target: True labels.
        epsilon: Perturbation magnitude.
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
    Projected Gradient Descent (PGD) attack (Iterative FGSM).
    
    Args:
        model: The neural network model.
        data: Input tensor.
        target: True labels.
        epsilon: Maximum perturbation (L-inf norm).
        alpha: Step size per iteration.
        num_iter: Number of iterations.
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
