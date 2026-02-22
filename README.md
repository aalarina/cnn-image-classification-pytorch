# Artifact Detection in Generated Images 

## Overview
This repository presents a deep learning project for detecting artifacts in generated images using a Convolutional Neural Network (CNN).

Generated images may contain undesired artifacts such as text, distorted hands, tattoos, incorrectly oriented eyes, or fragments of masks. The dataset consists of images labeled as artifact (class 0) and clean (class 1), with a significant class imbalance (~1:9).

The goal is to develop a robust binary classifier capable of automatically identifying images containing such artifacts.

To achieve this, we implemented two models:

- A custom CNN designed from scratch

- A predefined ResNet-18 model for comparison

## Project Structure

scr/
├── dataset.py         # ArtifactDataset
├── models.py          # Model architectures: CNN and ResNet-18
├── train.py           # Training scripts and evaluation functions
├── utils/
    ├── metrics.py           # Utility functions (metrics, checkpoints)
    ├── seed.py
    ├── ---
experiments/
├── f1_cnn.md          # Training/validation/test results for CNN
├── f1_resnet18.md     # Training/validation/test results for ResNet-18
└── model_comparison.md # Side-by-side comparison of CNN vs ResNet-18






