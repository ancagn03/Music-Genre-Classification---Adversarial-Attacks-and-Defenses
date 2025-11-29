# Music Genre Classification - Adversarial Attacks and Defenses

This project explores the vulnerability of music genre classification models to adversarial attacks and evaluates the effectiveness of various defense mechanisms. Using the GTZAN dataset, we train deep learning models (CNN, ResNet18) and subject them to gradient-based attacks (FGSM, PGD) and optimization-based attacks (DeepFool). We then implement and test defenses like Adversarial Training and Feature Squeezing.

## 📂 Project Structure

```
project-root/
├── data/                 # GTZAN dataset folder
├── experiments/          # Experiment scripts
│   ├── evaluate_defenses.py
│   ├── run_attacks.py
│   ├── run_defenses.py
│   └── run_deepfool.py
├── notebooks/            # Jupyter notebooks for experiments
│   ├── check_gpu.ipynb
│   └── test_remote.ipynb
├── presentation/         # Presentation slides and assets
├── results/              # Saved models (.pth), CSV logs, and plots
├── src/
│   ├── attacks.py        # Implementation of FGSM, PGD
│   ├── dataset.py        # GTZAN data loading (Features & Spectrograms)
│   ├── defenses.py       # Feature Squeezing implementation
│   ├── models.py         # Model architectures (MLP, CNN, ResNet18)
│   ├── train.py          # Training loops (Standard & Adversarial)
│   └── utils.py          # Helper functions
├── visualization/        # Visualization scripts
│   ├── generate_deepfool_plot.py
│   ├── generate_plots.py
│   └── visualize_attacks.py
├── main.py               # Script for standard training
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## 🧠 Methodology

### Models
*   **Custom CNN**: A Convolutional Neural Network designed for processing Mel-spectrograms of audio clips.
*   **ResNet18**: A residual network architecture adapted for audio classification using transfer learning.
*   **MLP**: A simple Multi-Layer Perceptron baseline using extracted tabular features (MFCCs, spectral centroid, etc.).

### Adversarial Attacks
*   **FGSM (Fast Gradient Sign Method)**: A one-step attack that perturbs the input in the direction of the loss gradient to maximize error.
*   **PGD (Projected Gradient Descent)**: An iterative, stronger variant of FGSM that applies small perturbations multiple times while staying within a defined limit ($\epsilon$).
*   **DeepFool**: An untargeted iterative attack that computes the minimum perturbation required to cross the decision boundary of the classifier.

### Defenses
*   **Adversarial Training**: Retraining the model on a mixture of clean and adversarial examples to learn a more robust decision boundary.
*   **Feature Squeezing**: A pre-processing defense that reduces the bit depth of the input audio spectrograms to remove high-frequency adversarial noise before classification.

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd Music-Genre-Classification---Adversarial-Attacks-and-Defenses
pip install -r requirements.txt
```

### 2. Dataset Setup

Ensure the GTZAN dataset is located in the `data/` directory. The project supports both feature-based (CSV) and spectrogram-based loading.

## 🛠 Usage

### 1. Train Baseline Models

Train a standard model (MLP, CNN, or ResNet18) on the clean dataset.

```bash
# Train a custom CNN
python main.py --model cnn --epochs 100

# Train ResNet18 (Transfer Learning)
python main.py --model resnet18 --epochs 100
```

*Models are saved to the `results/` directory.*

### 2. Run Adversarial Attacks

Evaluate the robustness of trained models against FGSM and PGD attacks.

```bash
# Attack the best CNN model
python experiments/run_attacks.py --model cnn --model_path results/best_model_cnn.pth

# Attack ResNet18
python experiments/run_attacks.py --model resnet18 --model_path results/best_model_resnet18.pth
```

### 3. Train with Defenses (Adversarial Training)

Train models using Adversarial Training (Mixed or Pure) to improve robustness.

```bash
# Mixed Adversarial Training (50% Clean, 50% Adversarial)
python experiments/run_defenses.py --model cnn --train_mode adv_mixed --epochs 50

# Pure Adversarial Training (100% Adversarial)
python experiments/run_defenses.py --model resnet18 --train_mode adv_pure --epochs 50
```

### 4. Evaluate Defenses

Evaluate the performance of defended models or apply inference-time defenses like Feature Squeezing.

```bash
# Evaluate Feature Squeezing on a standard model
python experiments/evaluate_defenses.py --model cnn --model_path results/best_model_cnn.pth --defense squeezing

# Evaluate an Adversarially Trained model
python experiments/evaluate_defenses.py --model cnn --model_path results/model_cnn_adv_mixed.pth --defense none
```

### 5. DeepFool Analysis

Run the DeepFool minimum-norm attack to measure the robustness distance ($L_2$ norm).

```bash
python experiments/run_deepfool.py --model cnn --model_path results/best_model_cnn.pth
```

### 6. Visualization

Generate plots and spectrogram visualizations for the report/presentation.

```bash
# Generate attack performance plots
python visualization/generate_plots.py

# Generate spectrogram images (Clean vs Attacked)
python visualization/visualize_attacks.py
```

## 📊 Key Results

*   **Vulnerability**: Standard CNNs and ResNet18 models drop to near 0% accuracy under strong PGD attacks ($\epsilon=0.1$).
*   **Adversarial Training**: Mixed Adversarial Training proved to be the most effective defense, recovering significant accuracy (e.g., ~20% to ~68% under attack) with a minor trade-off in clean accuracy.
*   **Feature Squeezing**: Provides a lightweight inference defense but is less effective than adversarial training against iterative attacks like PGD.
*   **DeepFool**: Defended models require significantly larger perturbations (up to 37x higher $L_2$ norm) to be fooled compared to baseline models.

## 👥 Authors

*   **Bogdan George Carp**
*   **Anca-Maria Găină**

## 🏫 Institution

National University of Science and Technology POLITEHNICA Bucharest
Faculty of Electronics, Telecommunications and Information Technology
