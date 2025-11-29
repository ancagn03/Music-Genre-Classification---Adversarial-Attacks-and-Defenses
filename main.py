"""
Main Entry Point for Music Genre Classification Training.

This script handles the standard training of baseline models (MLP, CNN, ResNet18)
on the GTZAN dataset. It supports:
- Data loading (Features or Spectrograms).
- Model initialization.
- Training loop with validation.
- Learning rate scheduling and early stopping.
- Model checkpointing.

Usage:
    python main.py --model cnn --epochs 100
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
    """
    Main execution function.
    Parses command-line arguments and orchestrates the training process.
    """
    parser = argparse.ArgumentParser(description="Music Genre Classification - Baseline Training")
    parser.add_argument('--epochs', type=int, default=50, help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'cnn', 'resnet18'], help='Model architecture to train')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 regularization factor')
    args = parser.parse_args()

    # 1. Setup Environment
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
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    print(f"Model initialized: {model.__class__.__name__}")
    
    # 4. Training Loop
    best_val_acc = 0.0
    save_path = f'results/best_model_{args.model}.pth'
    patience_counter = 0
    
    print(f"Starting training... (Saving to {save_path})")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Step the scheduler based on validation accuracy
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print("  --> New best model saved!")
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
