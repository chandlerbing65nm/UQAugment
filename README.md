# UQFishAugment

A deep learning framework for fish sound classification with uncertainty quantification and advanced data augmentation techniques.

## Overview

This project implements a comprehensive audio classification system specifically designed for fish sound analysis. It incorporates uncertainty quantification (UQ) methods and sophisticated data augmentation techniques to improve model robustness and reliability.

## Features

- **Audio Classification**: Deep learning models for fish sound classification
- **Uncertainty Quantification**: Both epistemic and aleatoric uncertainty estimation
- **Data Augmentation**: Advanced audio augmentation techniques
- **Model Support**: Multiple model architectures including:
  - PANNs
  - AST (Audio Spectrogram Transformer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/chandlerbing65nm/UQFishAugment.git
cd UQFishAugment
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: PyTorch installation requires ROCm support:
```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/rocm5.6
```

## Project Structure

```
UQFishAugment/
├── config/           # Configuration files
├── datasets/         # Dataset implementations
├── methods/          # Model implementations
├── transforms/       # Audio transformations
├── specaug/         # Spectrogram augmentation
├── losses/          # Loss functions
├── loggers/         # Logging utilities
├── figures/         # Visualization outputs
├── probs_epistemic/ # Epistemic uncertainty outputs
├── probs_aleatoric/ # Aleatoric uncertainty outputs
└── logs/            # Training logs
```

## Usage

### Training

To train a model:
```bash
bash submit_train.sh
```

### Testing

To evaluate a model:
```bash
bash test.sh
```

### Feature Extraction

To extract features:
```bash
bash submit_extract.sh
```

### Visualization

To generate optimization visualizations:
```bash
python visualize_opt.py --dataset [dataset_name]
```

## Requirements

- Python 3.x
- PyTorch 2.2.0
- ROCm-compatible GPU
- Additional dependencies listed in `requirements.txt`