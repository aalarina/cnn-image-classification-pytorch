# Artifact Detection in Generated Images

This repository contains a deep learning project to detect artifacts in generated images. The dataset contains images labeled with artifacts (class 0) and clean images (class 1). The dataset is highly imbalanced (~1:9 ratio).

## Features
- Data preprocessing and augmentation for imbalanced datasets.
- Custom CNN model (≥3 conv layers) and ResNet-18 model.
- Training with F1 metric and class-weighted loss.
- Visualization of augmented examples.
- Confusion matrix reporting for train/validation/test.
- Fully modular PyTorch pipeline.
