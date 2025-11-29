"""
Defense Evaluation Script.

This script evaluates the robustness of trained models (standard or adversarially trained)
against adversarial attacks (FGSM, PGD) with optional input transformation defenses (Feature Squeezing).

It computes:
- Clean Accuracy: Accuracy on original test data.
- Adversarial Accuracy: Accuracy on adversarial examples generated from test data.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.models import MusicMLP, MusicCNN, MusicResNet18
from src.dataset import GTZANDataset, GTZANSpectrogramDataset
from src.attacks import fgsm_attack, pgd_attack
from src.defenses import FeatureSqueezing
from src.utils import set_seed, get_device

def evaluate_defense(model, loader, attack_func, epsilon, device, defense_func=None, attack_name="FGSM", pgd_steps=10):
    """
    Evaluates the model accuracy under a specific attack, optionally applying a defense.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Test data loader.
        attack_func (callable): Function to generate adversarial examples.
        epsilon (float): Perturbation magnitude.
        device (torch.device): Computation device.
        defense_func (callable, optional): Defense function applied to input (e.g., Feature Squeezing).
        attack_name (str): Name of the attack for logging.
        pgd_steps (int): Number of steps for PGD attack.

    Returns:
        float: Accuracy percentage.
    """
    model.eval()
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(loader, desc=f"Eval {attack_name} (eps={epsilon})", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 1. Generate Adversarial Examples
        # Note: We generate attacks on the model directly. 
        # If the model was trained with squeezing, it might be robust naturally.
        # If we are testing inference defense, we attack, then squeeze.
        if epsilon == 0:
            adv_inputs = inputs
        else:
            if attack_name == "PGD":
                alpha = epsilon / 4
                adv_inputs = attack_func(model, inputs, labels, epsilon, alpha=alpha, num_iter=pgd_steps)
            else:
                adv_inputs = attack_func(model, inputs, labels, epsilon)
        
        # 2. Apply Defense (if any)
        if defense_func:
            adv_inputs = defense_func(adv_inputs)
            
        # 3. Evaluate
        with torch.no_grad():
            outputs = model(adv_inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    acc = 100. * correct / total
    return acc

def main():
    """
    Main execution function for evaluating defenses.
    """
    parser = argparse.ArgumentParser(description="Evaluate Defenses against Attacks")
    parser.add_argument("--model", type=str, required=True, choices=["mlp", "cnn", "resnet18"], help="Model architecture")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--defense", type=str, default="none", choices=["none", "squeezing"], help="Test time defense")
    parser.add_argument("--bit_depth", type=int, default=5, help="Bit depth for Feature Squeezing")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    
    # Load Data
    if args.model == "mlp":
        test_dataset = GTZANDataset(split='test', seed=args.seed)
        model = MusicMLP().to(device)
    elif args.model == "cnn":
        test_dataset = GTZANSpectrogramDataset(split='test', seed=args.seed)
        model = MusicCNN().to(device)
    elif args.model == "resnet18":
        test_dataset = GTZANSpectrogramDataset(split='test', seed=args.seed)
        model = MusicResNet18().to(device)
        
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load Weights
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Error: Model path {args.model_path} not found.")
        return

    # Setup Defense
    defense_func = None
    if args.defense == "squeezing":
        print(f"Defense Enabled: Feature Squeezing (Bit Depth: {args.bit_depth})")
        squeezer = FeatureSqueezing(bit_depth=args.bit_depth)
        defense_func = lambda x: squeezer(x)

    # Define Attacks to Evaluate
    epsilons = [0.0, 0.01, 0.03, 0.1] # Standard epsilons
    results = []
    
    print(f"Evaluating {args.model} with defense={args.defense}...")
    
    for eps in epsilons:
        # Clean / FGSM
        attack_name = "Clean" if eps == 0 else "FGSM"
        acc = evaluate_defense(model, test_loader, fgsm_attack, eps, device, defense_func, attack_name=attack_name)
        print(f"  {attack_name} (eps={eps}): {acc:.2f}%")
        results.append({"Attack": attack_name, "Epsilon": eps, "Accuracy": acc})
        
        # PGD (only for eps > 0)
        if eps > 0:
            acc_pgd = evaluate_defense(model, test_loader, pgd_attack, eps, device, defense_func, attack_name="PGD", pgd_steps=20)
            print(f"  PGD (eps={eps}): {acc_pgd:.2f}%")
            results.append({"Attack": "PGD", "Epsilon": eps, "Accuracy": acc_pgd})

    # Save Results
    df = pd.DataFrame(results)
    model_name = os.path.basename(args.model_path).replace(".pth", "")
    output_file = f"results/eval_{model_name}_def_{args.defense}.csv"
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
