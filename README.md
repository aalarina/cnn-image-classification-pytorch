# Artifact Detection in Generated Images 

**Dataset:** [Google Drive link](https://drive.google.com/file/d/16bv-5qEL_ajlZ1FbdSjJMRXcg3LS4lzb/view?usp=sharing)

## Overview
This repository presents a deep learning project for detecting artifacts in generated images using a Convolutional Neural Network (CNN).

Generated images may contain undesired artifacts such as text, distorted hands, tattoos, incorrectly oriented eyes, or fragments of masks. The dataset consists of images labeled as artifact (class 0) and clean (class 1), with a significant class imbalance (~1:9).

The goal is to develop a robust binary classifier capable of automatically identifying images containing such artifacts.

To achieve this, we implemented two models:

- A custom CNN designed from scratch

- A predefined ResNet-18 model for comparison

## Setup

### 1. Clone the repository

```git clone <repo-link>```

```cd <repo-name>```

### 2. Install dependencies (Python 3.9+ recommended)

```pip install torch torchvision scikit-learn matplotlib pyyaml pillow```

### 3. Prepare dataset

Place your dataset in ```dataset/ folder```:

```
dataset/
├── train/
└── test/
```

File naming convention: ```image_<index>_<label>.png``` (e.g., ```image_001_0.png```)

## Configuration

The training settings are stored in ```configs/config.yaml```:

```
batch_size: 32
learning_rate: 0.001
epochs: 10
image_size: 224
num_classes: 2
device: cuda
show_examples: True
```

## Features

**Class-specific augmentations** to handle imbalanced datasets.

**Train/Validation split** with stratification.

**CNN architecture** implemented from scratch and option to use ResNet18.

**Visualization** of augmented training examples.

**Weighted Cross-Entropy Loss** for class imbalance.

**Checkpointing** to save the best model automatically.

**Experiment logging:** all models and results saved under ```experiments/```.

## Results

After training, the script outputs:

- Training & validation loss per epoch

- F1 score per epoch

- Confusion matrix on train, validation, and test sets

Saved models and metrics are in ```experiments/``` for reproducibility.
