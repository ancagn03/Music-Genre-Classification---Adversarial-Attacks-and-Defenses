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
from src.dataset import GTZANDataset, GTZANSpectrogramDataset
from src.models import MusicMLP, MusicCNN, MusicResNet18
from src.train import train_one_epoch, validate

def main():
    parser = argparse.ArgumentParser(description="Music Genre Classification - Adversarial Attacks & Defenses")
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn', 'resnet18'], help='Model type: mlp, cnn, or resnet18')
    args = parser.parse_args()

    # 1. Setup
    set_seed(args.seed)
    device = get_device()
    
    # 2. Data Loading
    print(f"Loading data for {args.model.upper()} model...")
    
    if args.model == 'mlp':
        DatasetClass = GTZANDataset
        input_dim = 44
    else:
        DatasetClass = GTZANSpectrogramDataset
        
    train_dataset = DatasetClass(split='train', seed=args.seed)
    val_dataset = DatasetClass(split='val', seed=args.seed)
    test_dataset = DatasetClass(split='test', seed=args.seed)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 3. Model Initialization
    if args.model == 'mlp':
        model = MusicMLP(input_dim=input_dim, num_classes=10).to(device)
    elif args.model == 'cnn':
        model = MusicCNN(num_classes=10).to(device)
    elif args.model == 'resnet18':
        model = MusicResNet18(num_classes=10).to(device)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Model initialized: {model.__class__.__name__}")
    
    # 4. Training Loop
    best_val_acc = 0.0
    save_path = f'results/best_model_{args.model}.pth'
    
    print(f"Starting training... (Saving to {save_path})")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print("  --> New best model saved!")

    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
