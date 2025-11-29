"""
DeepFool Robustness Visualization.

This script generates a bar chart comparing the robustness of different models
against the DeepFool attack. Robustness is measured by the average L2 norm
of the perturbation required to change the model's prediction.
Higher L2 norm indicates a more robust model (decision boundary is further away).
"""
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_deepfool_comparison():
    """
    Generates and saves a bar chart comparing DeepFool robustness across models.
    """
    # Data
    models = ['CNN', 'ResNet18']
    
    # Average L2 Norm required to fool the model
    # Baseline values (Standard Training)
    baseline_l2 = [0.55, 0.55] 
    
    # Defended values (Adversarial Training)
    # CNN: Pure=34.43, Mixed=20.32
    # ResNet: Pure=6.36, Mixed=4.85
    defended_pure_l2 = [34.43, 6.36]
    defended_mixed_l2 = [20.32, 4.85]
    
    x = np.arange(len(models))
    width = 0.25
    
    plt.figure(figsize=(10, 6))
    
    # Plot bars
    rects1 = plt.bar(x - width, baseline_l2, width, label='Baseline', color='tab:red', alpha=0.7)
    rects2 = plt.bar(x, defended_mixed_l2, width, label='Adv. Training (Mixed)', color='tab:blue', alpha=0.7)
    rects3 = plt.bar(x + width, defended_pure_l2, width, label='Adv. Training (Pure)', color='tab:green', alpha=0.7)
    
    plt.ylabel('Avg. L2 Norm to Fool (Robustness)', fontsize=12)
    plt.title('DeepFool Robustness: Distance to Decision Boundary', fontsize=14)
    plt.xticks(x, models, fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs('presentation', exist_ok=True)
    
    plt.savefig('presentation/plot_deepfool_robustness.png')
    print("Saved presentation/plot_deepfool_robustness.png")


if __name__ == "__main__":
    os.makedirs('presentation', exist_ok=True)
    plot_deepfool_comparison()
