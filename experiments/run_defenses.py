"""
Adversarial Training Script.

This script trains models using Adversarial Training (Madry et al.) to improve robustness.
It supports:
- Standard Training (Baseline).
- Pure Adversarial Training (100% adversarial examples).
- Mixed Adversarial Training (50% clean, 50% adversarial).
- Feature Squeezing (as a pre-processing defense during training).
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from src.models import MusicMLP, MusicCNN, MusicResNet18
from src.dataset import GTZANDataset, GTZANSpectrogramDataset
from torch.utils.data import DataLoader
from src.train import train_one_epoch, train_adversarial_epoch, validate
from src.defenses import FeatureSqueezing
from src.utils import set_seed
import os

def main():
    """
    Main execution function for training with defenses.
    """
    parser = argparse.ArgumentParser(description="Train models with Defenses (Adversarial Training & Feature Squeezing)")
    parser.add_argument("--model", type=str, required=True, choices=["mlp", "cnn", "resnet18"], help="Model architecture")
    parser.add_argument("--defense", type=str, default="none", choices=["none", "squeezing"], help="Apply Feature Squeezing defense")
    parser.add_argument("--train_mode", type=str, default="standard", choices=["standard", "adv_pure", "adv_mixed"], help="Training mode: standard, adv_pure (100% adv), adv_mixed (50/50)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--bit_depth", type=int, default=5, help="Bit depth for Feature Squeezing")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Epsilon for Adversarial Training attacks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    if args.model == "mlp":
        train_dataset = GTZANDataset(split='train', seed=args.seed)
        val_dataset = GTZANDataset(split='val', seed=args.seed)
        test_dataset = GTZANDataset(split='test', seed=args.seed)
    else:
        train_dataset = GTZANSpectrogramDataset(split='train', seed=args.seed)
        val_dataset = GTZANSpectrogramDataset(split='val', seed=args.seed)
        test_dataset = GTZANSpectrogramDataset(split='test', seed=args.seed)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize Model
    if args.model == "mlp":
        model = MusicMLP().to(device)
    elif args.model == "cnn":
        model = MusicCNN().to(device)
    elif args.model == "resnet18":
        model = MusicResNet18().to(device)
    
    # Optimizer & Criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Setup Defense (Preprocessing)
    pre_process_func = None
    if args.defense == "squeezing":
        print(f"Defense: Feature Squeezing (Bit Depth: {args.bit_depth})")
        squeezer = FeatureSqueezing(bit_depth=args.bit_depth)
        pre_process_func = lambda x: squeezer(x)
    
    # Training Loop
    print(f"Starting training: Model={args.model}, Mode={args.train_mode}, Defense={args.defense}")
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        if args.train_mode == "standard":
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, 
                pre_process_func=pre_process_func
            )
        else:
            # Adversarial Training
            mix_ratio = 1.0 if args.train_mode == "adv_pure" else 0.5
            train_loss, train_acc = train_adversarial_epoch(
                model, train_loader, optimizer, criterion, device, 
                epsilon=args.epsilon, mix_ratio=mix_ratio, 
                pre_process_func=pre_process_func
            )
            
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            # Construct filename based on config
            defense_tag = f"_sq{args.bit_depth}" if args.defense == "squeezing" else ""
            mode_tag = f"_{args.train_mode}"
            save_path = f"results/model_{args.model}{mode_tag}{defense_tag}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    print("Training complete.")

if __name__ == "__main__":
    main()
