# Music-Genre-Classification---Adversarial-Attacks-and-Defenses
Categorizing music tracks based on their genre using audio features.

## Project Plan — Adversarial Attacks & Defenses on Music Genre Classification
1. Project Overview

### Objective:
Explore how machine learning models for music genre classification respond to adversarial perturbations and evaluate defense mechanisms that improve robustness.

### Main Dataset:
🎶 Million Song Dataset (MSD) – using precomputed feature vectors (main phase). http://millionsongdataset.com/pages/example-track-description/

Optional Extension: 
🎧 Cadenza / GTZAN / FMA audio datasets – for experiments on spectrograms or raw audio (later phase). https://zenodo.org/records/17252365

#### Core Attacks:

FGSM (Fast Gradient Sign Method) — simple one-step white-box attack

Minimum-Norm Attack (FMN/DDN-style) — iterative, minimal perturbation needed to flip label

#### Core Defenses:

Adversarial Training (PGD-based)

Feature Squeezing (Quantization)

## Proposed structure:

project-root/
├── data/                 # MSD features, splits, and optional audio subset
├── notebooks/            # Exploratory analysis and quick checks
├── src/
│   ├── models.py         # MLP model definitions (2–3 layers)
│   ├── train.py          # Clean + adversarial training loops
│   ├── attacks.py        # FGSM, Minimum-Norm, optional PGD
│   ├── defenses.py       # Adversarial Training, Feature Squeezing
│   ├── eval.py           # Evaluation, metrics, confusion, plotting
│   └── utils.py          # Dataset loading, normalization, seed control
├── experiments/          # Config files (YAML/JSON) for runs
├── results/              # Logs, saved models, plots, adversarial samples
├── presentation/         # Slides, report, figures
└── README.md             # Project summary and run instructions


Milestones & minimal timeline

### Milestone A — Setup & baseline

    Download / prepare MSD feature data.

    Create train/val/test splits (fixed seed).

    Implement simple MLP (2 hidden layers) + training loop.

    Produce baseline clean accuracy + confusion matrix.

### Milestone B — Implement attacks

    Implement FGSM (L∞) and Minimum-Norm (L₂, FMN/DDN-style).

    Run attacks on test set and report robust accuracy, ASR, and average L₂.

    Plot example feature perturbations (histograms).

### Milestone C — Implement defenses 

    Implement adversarial training (PGD-based recipe) — small epoch budget.

    Implement feature-squeezing (quantize to n decimals / reduce precision).

    Re-evaluate defenses against FGSM / Minimum-Norm / PGD (same hyperparams).

### Milestone D — Analysis & presentation

    Tables/plots: clean vs attacked vs defended.

    Per-class robustness and confusion matrices.

    Short slide deck + README + code tidy.

### (Optional) Milestone E — Audio Extension

    Use Cadenza / GTZAN / FMA dataset (with raw audio)

    Extract Mel-spectrograms for 100–200 clips

    Train small CNN or reuse MLP on spectrogram features

    Apply FGSM and Minimum-Norm on spectrograms

    Visualize:

        Original vs. adversarial spectrograms

        Audio difference (optional playback)

    Compare robustness between feature and audio domains