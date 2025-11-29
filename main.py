"""
Main entry point for the project.
Parses arguments and runs the selected experiment.
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils import set_seed, get_device
from src.dataset import GTZANDataset
from src.models import MusicMLP
from src.train import train_one_epoch, validate

def main():
    parser = argparse.ArgumentParser(description="Music Genre Classification - Adversarial Attacks & Defenses")
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # 1. Setup
    set_seed(args.seed)
    device = get_device()
    
    # 2. Data Loading
    print("Loading data...")
    train_dataset = GTZANDataset(split='train', seed=args.seed)
    val_dataset = GTZANDataset(split='val', seed=args.seed)
    test_dataset = GTZANDataset(split='test', seed=args.seed)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 3. Model Initialization
    model = MusicMLP(input_dim=44, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Model initialized: {model}")
    
    # 4. Training Loop
    best_val_acc = 0.0
    
    print("Starting training...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'results/best_model.pth')
            print("  --> New best model saved!")

    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
