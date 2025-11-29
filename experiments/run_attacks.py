"""
Adversarial Attack Evaluation Script.

This script evaluates the robustness of trained models against gradient-based attacks.
It supports:
- FGSM (Fast Gradient Sign Method)
- PGD (Projected Gradient Descent)

It iterates over a range of epsilon values (perturbation magnitudes) and logs the
accuracy degradation.
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

from src.utils import set_seed, get_device
from src.dataset import GTZANDataset, GTZANSpectrogramDataset
from src.models import MusicMLP, MusicCNN, MusicResNet18
from src.attacks import fgsm_attack, pgd_attack

def evaluate_attack(model, loader, attack_func, epsilon, device, attack_name="FGSM", pgd_steps=10, pgd_alpha=None):
    """
    Evaluates the model accuracy under a specific adversarial attack.
    
    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Test data loader.
        attack_func (callable): The attack function (fgsm_attack or pgd_attack).
        epsilon (float): The perturbation magnitude.
        device (torch.device): Computation device.
        attack_name (str): Name of the attack for logging.
        pgd_steps (int): Number of steps for PGD.
        pgd_alpha (float): Step size for PGD.
        
    Returns:
        tuple: (accuracy_percentage, elapsed_time_seconds)
    """
    model.eval()
    correct = 0
    total = 0
    
    start_time = time.time()
    
    # We iterate batch by batch
    for inputs, labels in tqdm(loader, desc=f"Running {attack_name} (eps={epsilon})", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Generate adversarial examples
        if epsilon == 0:
            adv_inputs = inputs # Clean data
        else:
            # PGD might need extra args like alpha/num_iter, handling simple case here
            if attack_name == "PGD":
                # Common heuristic: alpha = epsilon / 4 if not provided
                alpha = pgd_alpha if pgd_alpha is not None else epsilon / 4
                adv_inputs = attack_func(model, inputs, labels, epsilon, alpha=alpha, num_iter=pgd_steps)
            else:
                adv_inputs = attack_func(model, inputs, labels, epsilon)
        
        # Evaluate on adversarial examples
        with torch.no_grad():
            outputs = model(adv_inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    acc = 100. * correct / total
    return acc, elapsed_time

def main():
    """
    Main execution function for attack evaluation.
    """
    parser = argparse.ArgumentParser(description="Run Adversarial Attacks")
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn', 'resnet18'], help='Model to attack')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--steps', type=int, default=10, help='PGD iterations (default: 10)')
    parser.add_argument('--alpha', type=float, default=None, help='PGD step size (default: eps/4)')
    parser.add_argument('--output', type=str, default=None, help='Custom output CSV filename')
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
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. Load Model
    print(f"Loading trained {args.model.upper()} model...")
    if args.model == 'mlp':
        model = MusicMLP(input_dim=input_dim, num_classes=10).to(device)
    elif args.model == 'cnn':
        model = MusicCNN(num_classes=10).to(device)
    elif args.model == 'resnet18':
        model = MusicResNet18(num_classes=10).to(device)
        
    model_path = f'results/best_model_{args.model}.pth'
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Model file not found at {model_path}. Please train the model first.")
        return

    # 3. Define Attack Parameters
    # Epsilon represents the "strength" of the noise.
    # For audio/spectrograms, values usually range from 0.001 to 0.1 or higher depending on normalization.
    epsilons = [0, 0.001, 0.005, 0.01, 0.05, 0.1]
    
    results = []

    # 4. Run Attacks
    print("\n--- Starting Attack Evaluation ---")
    for eps in epsilons:
        # FGSM
        acc_fgsm, time_fgsm = evaluate_attack(model, test_loader, fgsm_attack, eps, device, attack_name="FGSM")
        
        # PGD (Skip PGD for eps=0 as it's same as clean)
        if eps == 0:
            acc_pgd = acc_fgsm
            time_pgd = time_fgsm
        else:
            acc_pgd, time_pgd = evaluate_attack(model, test_loader, pgd_attack, eps, device, attack_name="PGD", pgd_steps=args.steps, pgd_alpha=args.alpha)
            
        print(f"Epsilon: {eps:<6} | FGSM Acc: {acc_fgsm:.2f}% ({time_fgsm:.2f}s) | PGD Acc: {acc_pgd:.2f}% ({time_pgd:.2f}s)")
        
        results.append({
            'epsilon': eps,
            'fgsm_acc': acc_fgsm,
            'fgsm_time': time_fgsm,
            'pgd_acc': acc_pgd,
            'pgd_time': time_pgd
        })

    # 5. Save Results
    df = pd.DataFrame(results)
    if args.output:
        save_file = args.output
    else:
        save_file = f'results/attack_results_{args.model}.csv'
        
    df.to_csv(save_file, index=False)
    print(f"\nResults saved to {save_file}")

if __name__ == "__main__":
    main()
