"""
Attack Visualization Script.

This script generates visual comparisons of original and adversarial spectrograms.
It visualizes:
1. Original Spectrogram (Clean).
2. Adversarial Perturbation (Noise).
3. Adversarial Spectrogram (Attacked).

It supports FGSM, PGD, and DeepFool attacks.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import librosa.display
from src.models import MusicCNN
from src.dataset import GTZANSpectrogramDataset
from src.attacks import fgsm_attack, pgd_attack
from src.utils import get_device

# DeepFool Implementation (Simplified for visualization)
def deepfool_attack_viz(model, image, num_classes=10, overshoot=0.02, max_iter=50):
    """
    Simplified DeepFool implementation for visualization purposes.
    
    Args:
        model (nn.Module): The classifier.
        image (torch.Tensor): Input image.
        num_classes (int): Number of classes.
        overshoot (float): Overshoot parameter.
        max_iter (int): Maximum iterations.
        
    Returns:
        tuple: (perturbed_image, perturbation)
    """
    image = image.clone().detach()
    image.requires_grad = True
    
    output = model(image)
    _, pred = torch.max(output, 1)
    
    input_shape = image.shape
    pert_image = image.clone()
    w = torch.zeros(input_shape).to(image.device)
    r_tot = torch.zeros(input_shape).to(image.device)
    
    loop_i = 0
    x = pert_image.clone().detach()
    x.requires_grad = True
    
    fs = model(x)
    k_i = fs.data.argmax()
    
    while k_i == pred and loop_i < max_iter:
        fs[0, k_i].backward(retain_graph=True)
        grad_orig = x.grad.data.clone()
        
        pert = float('inf')
        w_best = None
        
        for k in range(num_classes):
            if k == k_i: continue
            
            x.grad.zero_()
            fs[0, k].backward(retain_graph=True)
            grad_current = x.grad.data.clone()
            
            w_k = grad_current - grad_orig
            f_k = (fs[0, k] - fs[0, k_i]).data
            
            pert_k = abs(f_k) / w_k.norm()
            
            if pert_k < pert:
                pert = pert_k
                w_best = w_k
        
        r_i = (pert + 1e-4) * w_best / w_best.norm()
        r_tot = r_tot + r_i
        
        x = image + (1 + overshoot) * r_tot
        x = x.clone().detach().requires_grad_(True)
        fs = model(x)
        k_i = fs.data.argmax()
        
        loop_i += 1
        
    return x, r_tot

def visualize_spectrogram(tensor, title, ax):
    """
    Helper to plot a spectrogram tensor.
    """
    # Tensor shape: (1, 1, H, W) -> (H, W)
    spec = tensor.squeeze().cpu().detach().numpy()
    img = librosa.display.specshow(spec, y_axis='mel', x_axis='time', ax=ax)
    ax.set_title(title)
    return img

def main():
    """
    Main execution function for generating attack visualizations.
    """
    device = get_device()
    
    # Load Model (CNN)
    model = MusicCNN().to(device)
    model_path = "results/best_model_cnn.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Model not found, using random weights for demo.")
        
    model.eval()
    
    # Load One Sample
    dataset = GTZANSpectrogramDataset(split='test')
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    image, label = next(iter(loader))
    image, label = image.to(device), label.to(device)
    
    # Get Prediction
    output = model(image)
    pred = output.argmax(dim=1).item()
    print(f"Original Prediction: {pred}, True Label: {label.item()}")
    
    # Generate Attacks
    # 1. FGSM
    adv_fgsm = fgsm_attack(model, image, label, epsilon=0.05)
    pert_fgsm = adv_fgsm - image
    
    # 2. PGD
    adv_pgd = pgd_attack(model, image, label, epsilon=0.05, alpha=0.01, num_iter=20)
    pert_pgd = adv_pgd - image
    
    # 3. DeepFool
    adv_df, pert_df = deepfool_attack_viz(model, image)
    
    # Plotting
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Row 1: FGSM
    visualize_spectrogram(image, "Original", axes[0, 0])
    visualize_spectrogram(pert_fgsm, "FGSM Perturbation (x10)", axes[0, 1]) # Scale for visibility?
    visualize_spectrogram(adv_fgsm, "FGSM Attack", axes[0, 2])
    
    # Row 2: PGD
    visualize_spectrogram(image, "Original", axes[1, 0])
    visualize_spectrogram(pert_pgd, "PGD Perturbation", axes[1, 1])
    visualize_spectrogram(adv_pgd, "PGD Attack", axes[1, 2])
    
    # Row 3: DeepFool
    visualize_spectrogram(image, "Original", axes[2, 0])
    visualize_spectrogram(pert_df, "DeepFool Perturbation", axes[2, 1])
    visualize_spectrogram(adv_df, "DeepFool Attack", axes[2, 2])
    
    plt.tight_layout()
    os.makedirs("presentation", exist_ok=True)
    plt.savefig("presentation/attack_visualization.png")
    print("Saved presentation/attack_visualization.png")

if __name__ == "__main__":
    main()
            fs[0, k].backward(retain_graph=True)
            cur_grad = x.grad.data.clone()
            
            w_k = cur_grad - grad_orig
            f_k = (fs[0, k] - fs[0, k_i]).data
            
            pert_k = abs(f_k) / torch.norm(w_k.flatten())
            
            if pert_k < pert:
                pert = pert_k
                w_best = w_k
                
        r_i = (pert + 1e-4) * w_best / torch.norm(w_best)
        r_tot = r_tot + r_i
        
        x = image + (1 + overshoot) * r_tot
        x = x.clone().detach().requires_grad_(True)
        fs = model(x)
        k_i = fs.data.argmax()
        loop_i += 1
        
    return (1 + overshoot) * r_tot

def visualize_attacks():
    device = get_device()
    
    # Load Model (CNN Baseline)
    model = MusicCNN().to(device)
    # We need a baseline model, let's assume best_model_cnn.pth exists or use one of the results
    # Ideally we use the undefended one to show vulnerability
    model_path = 'results/best_model_cnn.pth' 
    if not os.path.exists(model_path):
        # Fallback to one of the adv ones if baseline is missing, but baseline is better
        model_path = 'results/model_cnn_adv_pure.pth'
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load One Sample
    dataset = GTZANSpectrogramDataset(split='test')
    # Pick a sample that is correctly classified
    idx = 0
    while True:
        img, label = dataset[idx]
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img).argmax().item()
        if pred == label:
            break
        idx += 1
        
    print(f"Visualizing sample {idx}, Class: {dataset.classes[label]}")
    
    # Generate Attacks
    # 1. FGSM
    epsilon = 0.05
    delta_fgsm = fgsm_attack(model, img, torch.tensor([label]).to(device), epsilon) - img
    
    # 2. PGD
    delta_pgd = pgd_attack(model, img, torch.tensor([label]).to(device), epsilon, alpha=0.01, num_iter=20) - img
    
    # 3. DeepFool
    # Note: DeepFool returns the perturbation r_tot
    r_tot = deepfool_attack_viz(model, img)
    delta_df = r_tot
    
    # Plotting
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    
    # Helper to plot spectrogram
    def plot_spec(ax, data, title):
        # data is (1, 128, 128) tensor
        d = data.squeeze().cpu().detach().numpy()
        im = ax.imshow(d, origin='lower', aspect='auto', cmap='magma')
        ax.set_title(title)
        ax.axis('off')
        return im

    # Row 1: Original
    plot_spec(axes[0, 0], img, f"Original\n(Pred: {dataset.classes[label]})")
    axes[0, 1].axis('off') # No noise for original
    axes[0, 2].axis('off')
    
    # Row 2: FGSM
    adv_fgsm = img + delta_fgsm
    pred_fgsm = model(adv_fgsm).argmax().item()
    plot_spec(axes[1, 0], img, "Original")
    plot_spec(axes[1, 1], delta_fgsm * 10, "FGSM Noise (x10)") # Amplify noise for visibility
    plot_spec(axes[1, 2], adv_fgsm, f"FGSM Attack\n(Pred: {dataset.classes[pred_fgsm]})")
    
    # Row 3: PGD
    adv_pgd = img + delta_pgd
    pred_pgd = model(adv_pgd).argmax().item()
    plot_spec(axes[2, 0], img, "Original")
    plot_spec(axes[2, 1], delta_pgd * 10, "PGD Noise (x10)")
    plot_spec(axes[2, 2], adv_pgd, f"PGD Attack\n(Pred: {dataset.classes[pred_pgd]})")
    
    # Row 4: DeepFool
    adv_df = img + delta_df
    pred_df = model(adv_df).argmax().item()
    plot_spec(axes[3, 0], img, "Original")
    plot_spec(axes[3, 1], delta_df * 10, "DeepFool Noise (x10)")
    plot_spec(axes[3, 2], adv_df, f"DeepFool Attack\n(Pred: {dataset.classes[pred_df]})")
    
    plt.tight_layout()
    plt.savefig('presentation/spectrogram_attacks.png')
    print("Saved presentation/spectrogram_attacks.png")

    # --- Generate Separate Images ---
    
    # 1. FGSM Only
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plot_spec(axes[0], img, f"Original\n(Pred: {dataset.classes[label]})")
    plot_spec(axes[1], delta_fgsm * 10, "FGSM Noise (x10)")
    plot_spec(axes[2], adv_fgsm, f"FGSM Attack\n(Pred: {dataset.classes[pred_fgsm]})")
    plt.tight_layout()
    plt.savefig('presentation/spectrogram_fgsm.png')
    print("Saved presentation/spectrogram_fgsm.png")
    plt.close(fig)

    # 2. PGD Only
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plot_spec(axes[0], img, f"Original\n(Pred: {dataset.classes[label]})")
    plot_spec(axes[1], delta_pgd * 10, "PGD Noise (x10)")
    plot_spec(axes[2], adv_pgd, f"PGD Attack\n(Pred: {dataset.classes[pred_pgd]})")
    plt.tight_layout()
    plt.savefig('presentation/spectrogram_pgd.png')
    print("Saved presentation/spectrogram_pgd.png")
    plt.close(fig)

    # 3. DeepFool Only
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plot_spec(axes[0], img, f"Original\n(Pred: {dataset.classes[label]})")
    plot_spec(axes[1], delta_df * 10, "DeepFool Noise (x10)")
    plot_spec(axes[2], adv_df, f"DeepFool Attack\n(Pred: {dataset.classes[pred_df]})")
    plt.tight_layout()
    plt.savefig('presentation/spectrogram_deepfool.png')
    print("Saved presentation/spectrogram_deepfool.png")
    plt.close(fig)

if __name__ == "__main__":
    os.makedirs('presentation', exist_ok=True)
    visualize_attacks()
