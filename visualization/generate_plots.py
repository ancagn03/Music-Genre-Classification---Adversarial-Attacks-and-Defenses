"""
Plot Generation Script.

This script generates visualization plots for the presentation and report.
It reads the CSV results from the `results/` directory and produces:
1. Baseline Vulnerability Curves (Accuracy vs. Epsilon).
2. Defense Effectiveness Comparison (Baseline vs. Adversarial Training).
3. Feature Squeezing Impact Analysis.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_attack_curve():
    """
    Generates and saves plots comparing model robustness against PGD attacks.
    """
    # Load Baseline Results
    df_mlp = pd.read_csv('results/attack_results_mlp.csv')
    df_cnn = pd.read_csv('results/attack_results_cnn.csv')
    df_resnet = pd.read_csv('results/attack_results_resnet18.csv')
    
    # Helper to load defended results
    def get_defended_acc(model_name, attack_type='PGD'):
        """
        Retrieves accuracy vs epsilon for a defended model.
        """
        filename = f'results/eval_model_{model_name}_adv_mixed_def_none.csv'
        if not os.path.exists(filename):
            return None, None
        df = pd.read_csv(filename)
        # Filter for specific attack
        df_att = df[df['Attack'] == attack_type]
        return df_att['Epsilon'].values, df_att['Accuracy'].values

    # Plot 1: Baseline Vulnerability (PGD)
    plt.figure(figsize=(10, 6))
    plt.plot(df_mlp['epsilon'], df_mlp['pgd_acc'], marker='o', label='MLP (Baseline)', linewidth=2)
    plt.plot(df_cnn['epsilon'], df_cnn['pgd_acc'], marker='s', label='CNN (Baseline)', linewidth=2)
    plt.plot(df_resnet['epsilon'], df_resnet['pgd_acc'], marker='^', label='ResNet18 (Baseline)', linewidth=2)
    
    plt.title('Baseline Model Vulnerability to PGD Attack', fontsize=14)
    plt.xlabel('Perturbation Magnitude (Epsilon)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.ylim(0, 100)
    plt.savefig('presentation/plot_baseline_vulnerability.png')
    print("Saved presentation/plot_baseline_vulnerability.png")

    # Plot 2: Defense Effectiveness (CNN)
    eps_def, acc_def = get_defended_acc('cnn', 'PGD')
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_cnn['epsilon'], df_cnn['pgd_acc'], marker='s', label='CNN (Baseline)', color='tab:orange', linestyle='--', linewidth=2)
    if eps_def is not None:
        # Add 0.0 point (Clean accuracy) if not present in the filtered view
        # Note: The eval script usually saves Clean as eps=0 separately or as part of the list
        # We assume the helper returns what is in the file.
        
        # If the file structure separates Clean from PGD, we might need to fetch Clean separately.
        # Assuming the eval script saves PGD rows for eps > 0 and Clean for eps=0.
        
        # Let's check if 0 is in eps_def
        if 0.0 not in eps_def:
             # Try to find "Clean" row
             df_full = pd.read_csv(f'results/eval_model_cnn_adv_mixed_def_none.csv')
             clean_row = df_full[df_full['Attack'] == 'Clean']
             if not clean_row.empty:
                 clean_acc = clean_row['Accuracy'].values[0]
                 eps_def = np.insert(eps_def, 0, 0.0)
                 acc_def = np.insert(acc_def, 0, clean_acc)

        plt.plot(eps_def, acc_def, marker='o', label='CNN (Adv. Training)', color='tab:green', linewidth=2)
    
    plt.title('Defense Effectiveness: Adversarial Training (CNN)', fontsize=14)
    plt.xlabel('Perturbation Magnitude (Epsilon)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.ylim(0, 100)
    plt.savefig('presentation/plot_defense_effectiveness.png')
    print("Saved presentation/plot_defense_effectiveness.png")
        plt.plot(eps_full, acc_full, marker='o', label='CNN (Adv. Training)', color='tab:green', linewidth=2)

    plt.title('Defense Effectiveness: CNN under PGD Attack', fontsize=14)
    plt.xlabel('Perturbation Magnitude (Epsilon)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.ylim(0, 100)
    plt.savefig('presentation/plot_defense_effectiveness_cnn.png')
    print("Saved presentation/plot_defense_effectiveness_cnn.png")

def plot_model_comparison():
    # Data for Epsilon = 0.01 (PGD)
    models = ['MLP', 'CNN', 'ResNet18']
    
    # Baseline Accuracies at eps=0.01
    base_mlp = pd.read_csv('results/attack_results_mlp.csv').set_index('epsilon').loc[0.01, 'pgd_acc']
    base_cnn = pd.read_csv('results/attack_results_cnn.csv').set_index('epsilon').loc[0.01, 'pgd_acc']
    base_resnet = pd.read_csv('results/attack_results_resnet18.csv').set_index('epsilon').loc[0.01, 'pgd_acc']
    
    baseline_accs = [base_mlp, base_cnn, base_resnet]
    
    # Defended Accuracies at eps=0.01 (Adv Mixed)
    def get_acc_at_eps(model_name, eps=0.01):
        df = pd.read_csv(f'results/eval_model_{model_name}_adv_mixed_def_none.csv')
        row = df[(df['Attack'] == 'PGD') & (df['Epsilon'] == eps)]
        if not row.empty:
            return row.iloc[0]['Accuracy']
        return 0
        
    def_mlp = get_acc_at_eps('mlp')
    def_cnn = get_acc_at_eps('cnn')
    def_resnet = get_acc_at_eps('resnet18')
    
    defended_accs = [def_mlp, def_cnn, def_resnet]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, baseline_accs, width, label='Baseline (Undefended)', color='tab:red', alpha=0.7)
    plt.bar(x + width/2, defended_accs, width, label='Adversarial Training (Mixed)', color='tab:blue', alpha=0.7)
    
    plt.ylabel('Accuracy under PGD Attack (eps=0.01)', fontsize=12)
    plt.title('Robustness Comparison: Baseline vs. Defended Models', fontsize=14)
    plt.xticks(x, models, fontsize=12)
    plt.legend(fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add value labels
    for i, v in enumerate(baseline_accs):
        plt.text(i - width/2, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=10)
    for i, v in enumerate(defended_accs):
        plt.text(i + width/2, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=10)
        
    plt.savefig('presentation/plot_model_comparison_bar.png')
    print("Saved presentation/plot_model_comparison_bar.png")

def plot_all_attacks_overview():
    # Load Baseline Results
    df_mlp = pd.read_csv('results/attack_results_mlp.csv')
    df_cnn = pd.read_csv('results/attack_results_cnn.csv')
    df_resnet = pd.read_csv('results/attack_results_resnet18.csv')
    
    plt.figure(figsize=(12, 8))
    
    # MLP
    plt.plot(df_mlp['epsilon'], df_mlp['fgsm_acc'], marker='o', linestyle='-', color='tab:blue', label='MLP (FGSM)', alpha=0.7)
    plt.plot(df_mlp['epsilon'], df_mlp['pgd_acc'], marker='x', linestyle='--', color='tab:blue', label='MLP (PGD)', alpha=1.0, linewidth=2)
    
    # CNN
    plt.plot(df_cnn['epsilon'], df_cnn['fgsm_acc'], marker='o', linestyle='-', color='tab:orange', label='CNN (FGSM)', alpha=0.7)
    plt.plot(df_cnn['epsilon'], df_cnn['pgd_acc'], marker='x', linestyle='--', color='tab:orange', label='CNN (PGD)', alpha=1.0, linewidth=2)
    
    # ResNet18
    plt.plot(df_resnet['epsilon'], df_resnet['fgsm_acc'], marker='o', linestyle='-', color='tab:green', label='ResNet18 (FGSM)', alpha=0.7)
    plt.plot(df_resnet['epsilon'], df_resnet['pgd_acc'], marker='x', linestyle='--', color='tab:green', label='ResNet18 (PGD)', alpha=1.0, linewidth=2)
    
    plt.title('Vulnerability Overview: All Models & Attacks', fontsize=16)
    plt.xlabel('Perturbation Magnitude (Epsilon)', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.ylim(0, 100)
    
    plt.savefig('presentation/plot_all_attacks_overview.png')
    print("Saved presentation/plot_all_attacks_overview.png")

if __name__ == "__main__":
    os.makedirs('presentation', exist_ok=True)
    plot_attack_curve()
    plot_model_comparison()
    plot_all_attacks_overview()
