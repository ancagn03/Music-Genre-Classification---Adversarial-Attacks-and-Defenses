"""
DeepFool Attack Script.

This script implements the DeepFool algorithm (Moosavi-Dezfooli et al., 2016) to find the minimum
perturbation required to misclassify an input. It is an untargeted iterative attack that linearizes
the decision boundary at each step.

The perturbation is computed as:
$$
r_*(x) = \text{argmin}_{r} \|r\|_2 \quad \text{s.t.} \quad \text{sign}(f(x+r)) \neq \text{sign}(f(x))
$$

This script evaluates the robustness of models by measuring the average perturbation magnitude required to fool them.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import pandas as pd
from tqdm import tqdm
import time
import numpy as np

from src.utils import set_seed, get_device
from src.dataset import GTZANDataset, GTZANSpectrogramDataset
from src.models import MusicMLP, MusicCNN, MusicResNet18

def deepfool_attack(model, image, num_classes=10, overshoot=0.02, max_iter=50):
    """
    DeepFool Attack Implementation.
    
    Iteratively computes the minimum perturbation to cross the decision boundary of the classifier.
    
    Args:
        model (nn.Module): The neural network classifier.
        image (torch.Tensor): Single input image tensor of shape (1, C, H, W).
        num_classes (int): Number of output classes.
        overshoot (float): Small constant to ensure the boundary is crossed.
        max_iter (int): Maximum number of iterations.
        
    Returns:
        tuple: (perturbed_image, perturbation_tensor, iterations_used)
    """
    image = image.clone().detach()
    image.requires_grad = True
    
    output = model(image)
    # Get the predicted class
    _, pred_label = torch.max(output, 1)
    pred_label = pred_label.item()
    
    pert_image = image.clone()
    w = torch.zeros_like(image)
    r_tot = torch.zeros_like(image)
    
    loop_i = 0
    
    # Sort classes by probability (descending) to check closest boundaries first
    output_sorted_indices = output.data.argsort(descending=True)[0]
    
    while loop_i < max_iter:
        # Check if label changed
        output = model(pert_image)
        _, current_label = torch.max(output, 1)
        
        if current_label != pred_label:
            break
            
        # Compute gradient for the original predicted class
        model.zero_grad()
        
        # Detach to make it a leaf variable again, then enable grad for next step
        pert_image = pert_image.detach()
        pert_image.requires_grad = True
        output = model(pert_image)
        
        # Gradient of the original class (k_0)
        output[0, pred_label].backward(retain_graph=True)
        grad_orig = pert_image.grad.data.clone()
        
        pert = float('inf')
        w_best = torch.zeros_like(image)
        
        # Iterate over all other classes to find the closest decision boundary
        for k in range(1, num_classes):
            target_class = output_sorted_indices[k]
            
            model.zero_grad()
            output[0, target_class].backward(retain_graph=True)
            grad_current = pert_image.grad.data.clone()
            
            # w_k = grad_k - grad_0
            w_k = grad_current - grad_orig
            # f_k = f_k - f_0
            f_k = (output[0, target_class] - output[0, pred_label]).data
            
            # Distance to boundary k: |f_k| / ||w_k||_2
            w_norm = torch.norm(w_k, p=2)
            
            # Avoid division by zero
            if w_norm < 1e-6:
                continue
                
            dist_k = torch.abs(f_k) / w_norm
            
            if dist_k < pert:
                pert = dist_k
                w_best = w_k
                
        # Accumulate perturbation
        # r_i = (pert + 1e-4) * w_best / ||w_best||
        r_i = (pert + 1e-4) * w_best / torch.norm(w_best, p=2)
        r_tot = r_tot + r_i
        
        # Update image
        pert_image = image + (1 + overshoot) * r_tot
        loop_i += 1
        
    return pert_image, r_tot, loop_i

def main():
    parser = argparse.ArgumentParser(description="Run DeepFool (Minimum Norm) Attack")
    parser.add_argument('--model', type=str, default='cnn', choices=['mlp', 'cnn', 'resnet18'], help='Model to attack')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples to attack (DeepFool is slow)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    
    # 1. Load Data
    print(f"Loading test data for {args.model.upper()}...")
    if args.model == 'mlp':
        DatasetClass = GTZANDataset
        input_dim = 44
    else:
        DatasetClass = GTZANSpectrogramDataset
        
    test_dataset = DatasetClass(split='test', seed=args.seed)
    # Batch size 1 because DeepFool works per-image
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 2. Load Model
    print(f"Loading trained {args.model.upper()} model...")
    if args.model == 'mlp':
        model = MusicMLP(input_dim=input_dim, num_classes=10).to(device)
    elif args.model == 'cnn':
        model = MusicCNN(num_classes=10).to(device)
    elif args.model == 'resnet18':
        model = MusicResNet18(num_classes=10).to(device)
        
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = f'results/best_model_{args.model}.pth'
        
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Model weights loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"❌ Model file not found at {model_path}.")
        return

    model.eval()
    
    print(f"\n--- Starting DeepFool Attack on {args.samples} samples ---")
    
    success_count = 0
    total_l2_norm = 0.0
    processed_count = 0
    correct_clean_count = 0
    
    pbar = tqdm(total=args.samples)
    
    for i, (image, label) in enumerate(test_loader):
        if i >= args.samples:
            break
            
        image, label = image.to(device), label.to(device)
        
        # Check if correctly classified first
        output = model(image)
        _, pred = output.max(1)
        
        if pred != label:
            # Skip already misclassified images
            pbar.update(1)
            continue
            
        correct_clean_count += 1
            
        # Run DeepFool
        start_time = time.time()
        adv_image, perturbation, iters = deepfool_attack(model, image, num_classes=10)
        end_time = time.time()
        
        # Verify attack success
        adv_output = model(adv_image)
        _, adv_pred = adv_output.max(1)
        
        if adv_pred != label:
            success_count += 1
            l2_norm = torch.norm(perturbation, p=2).item()
            total_l2_norm += l2_norm
            processed_count += 1
            
        pbar.update(1)
        pbar.set_description(f"Avg L2: {total_l2_norm/processed_count if processed_count else 0:.4f}")
        
    pbar.close()
    
    if processed_count > 0:
        avg_l2 = total_l2_norm / processed_count
        attack_success_rate = 100. * success_count / correct_clean_count
        
        print(f"\nResults for {args.model.upper()}:")
        print(f"Total samples checked: {args.samples}")
        print(f"Correctly classified (Clean): {correct_clean_count}")
        print(f"Successfully attacked: {success_count}")
        print(f"Attack Success Rate: {attack_success_rate:.2f}%")
        print(f"Average L2 Norm required to fool: {avg_l2:.6f}")
        print(f"This represents the 'robustness' (distance to boundary). Lower is worse.")
    else:
        print("No correctly classified images found in the first N samples.")

if __name__ == "__main__":
    main()
