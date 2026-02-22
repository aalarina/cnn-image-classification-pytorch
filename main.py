import torch
import yaml

from models.model import get_model
from scr.dataset import ArtifactDataset
from data.transforms import get_transforms
from scr.train import run_training
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

def main():
    # Config
    config = load_config()

    # Seed
    set_seed(config.get("seed", 42))

    # Device
    device = torch.device(
        config["device"] if torch.cuda.is_available() else "cpu"
    )

    # Data
    train_loader, val_loader = prepare_dataloaders(config)

    # Model
    model = get_model(
        model_name=config["model_name"],   # "cnn" or "resnet18"
        num_classes=config["num_classes"]
    ).to(device)

    # Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"]
    )

    # Train
    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=config["epochs"]
    )


if __name__ == "__main__":
    main()
