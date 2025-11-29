"""
Training Loops for Standard and Adversarial Training.

This module contains the core training logic:
1. train_one_epoch: Standard training on clean data.
2. train_adversarial_epoch: Adversarial training using PGD-generated examples.
3. validate: Evaluation loop for validation/test sets.
"""
import torch
from tqdm import tqdm
from src.attacks import pgd_attack

def train_one_epoch(model, loader, optimizer, criterion, device, pre_process_func=None):
    """
    Trains the model for one epoch using standard supervised learning.
    
    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): PyTorch optimizer (e.g., Adam).
        criterion (Loss): Loss function (e.g., CrossEntropyLoss).
        device (torch.device): Device to run on (CPU/GPU).
        pre_process_func (callable, optional): Function to apply to inputs before forward pass (e.g., Feature Squeezing).
        
    Returns:
        tuple: (average_loss, accuracy_percentage)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(loader, desc="Training (Clean)", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        if pre_process_func:
            inputs = pre_process_func(inputs)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def train_adversarial_epoch(model, loader, optimizer, criterion, device, epsilon=0.01, mix_ratio=0.5, pre_process_func=None):
    """
    Trains the model using Adversarial Training (Madry et al.).
    
    This method generates adversarial examples on-the-fly using PGD and trains the model
    to correctly classify them. This is a Min-Max optimization problem:
    min_theta E [ max_delta L(theta, x + delta, y) ]
    
    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): PyTorch optimizer.
        criterion (Loss): Loss function.
        device (torch.device): Device to run on.
        epsilon (float): Maximum perturbation allowed for the inner attack.
        mix_ratio (float): Ratio of adversarial examples in the batch (0.0 = Clean, 1.0 = Pure Adv, 0.5 = Mixed).
        pre_process_func (callable, optional): Pre-processing defense to apply during training.
        
    Returns:
        tuple: (average_loss, accuracy_percentage)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(loader, desc=f"Training (Adv mix={mix_ratio})", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Apply preprocessing (e.g. Squeezing) BEFORE splitting/attacking
        # This simulates the defense being active during training
        if pre_process_func:
            inputs = pre_process_func(inputs)
        
        # Determine split size
        batch_size = inputs.size(0)
        num_adv = int(batch_size * mix_ratio)
        num_clean = batch_size - num_adv
        
        # Split batch
        inputs_clean = inputs[:num_clean]
        labels_clean = labels[:num_clean]
        
        inputs_to_attack = inputs[num_clean:]
        labels_to_attack = labels[num_clean:]
        
        # Generate Adversarial Examples (PGD is standard for Adv Training)
        # We use model.eval() context for attack generation to avoid BatchNorm updates during attack
        model.eval()
        if num_adv > 0:
            adv_inputs = pgd_attack(model, inputs_to_attack, labels_to_attack, epsilon=epsilon, alpha=epsilon/4, num_iter=7)
            model.train() # Switch back to train mode
        else:
            adv_inputs = torch.empty(0, *inputs.shape[1:]).to(device)
            model.train()
            
        # Combine Clean + Adversarial
        if num_clean > 0 and num_adv > 0:
            final_inputs = torch.cat([inputs_clean, adv_inputs])
            final_labels = torch.cat([labels_clean, labels_to_attack])
        elif num_adv > 0:
            final_inputs = adv_inputs
            final_labels = labels_to_attack
        else:
            final_inputs = inputs_clean
            final_labels = labels_clean
            
        # Standard Training Step
        optimizer.zero_grad()
        
        outputs = model(final_inputs)
        loss = criterion(outputs, final_labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += final_labels.size(0)
        correct += predicted.eq(final_labels).sum().item()
        
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    val_loss = running_loss / len(loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc
