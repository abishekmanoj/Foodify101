# Foodify101: Food Image Classification Using ResNet50

Foodify101 is a deep learning-based image classifier trained on the Food101 dataset using ResNet50 and progressive fine-tuning. It classifies images into 101 different food categories with strong performance and a modular training pipeline built in PyTorch.

## Overview

This project explores the application of transfer learning and layer-wise fine-tuning to adapt a pretrained ResNet50 model to the Food101 dataset. The training is performed in multiple phases to gradually improve performance without overfitting.

## Features

- Dataset: Food101 (downloaded via torchvision)
- Backbone: ResNet50 pretrained on ImageNet
- Modular code for training, evaluation, and visualization
- Progressive fine-tuning: fc only (Phase 1) → layer4 + fc (Phase 2) → layer3 + layer4 + fc (Phase 3)
- Layer-specific learning rates using AdamW (in Phase 3)
- Achieved over 81% test accuracy

## Project Structure

```
Foodify101/
├── data/             # Food101 dataset
├── models/           # Saved model checkpoints
├── modules/          # Training utilities
│   ├── engine.py     # Train/test loops
│   ├── data.py       # Dataset and dataloader creation
│   ├── utils.py      # Model saving, loading, summaries
│   └── visualize.py  # Loss curves and TensorBoard logging
├── train.ipynb       # Notebook for executing training phases
├── requirements.txt  # Project dependencies
├── experiments.md    # Training phase logs and results
└── README.md         # Project overview and instructions
```

## Training Phases

### Phase 1 - Head-Only Training
- Layers trained: final fc
- Learning rate: 1e-3
- Epochs: 15

### Phase 2 - Fine-Tuning layer4 + fc
- Added dropout to classification head
- Learning rate: 1e-4
- Epochs: 15

### Phase 3 - Progressive Fine-Tuning
- Layers trained: layer3, layer4, fc
- Learning rates:
  - layer3: 1e-6
  - layer4: 1e-5
  - fc: 1e-4
- Scheduler: StepLR (gamma=0.5 every 5 epochs)
- Epochs: 20
- Final test accuracy: 81.7%

## Installation

1. Clone the repository:
   - git clone https://github.com/abishekmanoj/foodify101.git
   - cd Foodify101
   
2. Install dependencies:
   - pip install -r requirements.txt
   
3. Run the train script:
   - Open and execute train.ipynb for step-by-step training phases.

## License
MIT License © 2025 Abishek Manoj