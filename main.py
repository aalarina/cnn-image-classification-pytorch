import torch
import yaml

from models.model import get_model
from data.dataset import ImageDataset
from data.transforms import get_transforms
from training.train import train
from utils.seed import set_seed

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def prepare_dataloaders(config):
    transform = get_transforms()

    train_dataset = ArtifactDataset(
        image_paths=config["train_paths"],
        labels=config["train_labels"],
        transform=transform
    )

    val_dataset = ArtifactDataset(
        image_paths=config["val_paths"],
        labels=config["val_labels"],
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False
    )

    return train_loader, val_loader
