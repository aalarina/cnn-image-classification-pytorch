import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from scr.dataset import ArtifactDataset
from scr.models import get_model
from utils.helpers import get_image_list_from_dir
from scr.train import train_one_epoch, validate, run_training

def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

train_files_labels = get_image_list_from_dir("dataset/train")
val_files_labels = get_image_list_from_dir("dataset/val")
test_files_labels = get_image_list_from_dir("dataset/test")

train_files, train_labels = zip(*train_files_labels)
val_files, val_labels = zip(*val_files_labels)
test_files, test_labels = zip(*test_files_labels)

train_dataset = ArtifactDataset(train_files, train_labels, train=True)
val_dataset = ArtifactDataset(val_files, val_labels, train=False)
test_dataset = ArtifactDataset(test_files, test_labels, train=False)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

model_cnn = get_model("cnn", num_classes=2)
model_cnn.to(config["device"])

labels_tensor = torch.tensor(train_labels)
class_counts = torch.bincount(labels_tensor)
class_weights = 1.0 / class_counts.float()
criterion = nn.CrossEntropyLoss(weight=class_weights.to(config["device"]))

model_cnn, history = run_training(model_cnn, train_loader, val_loader,
                                  epochs=config["epochs"], lr=config["learning_rate"],
                                  save_path="best_model_cnn.pth")

checkpoint_cnn = torch.load("best_model_cnn.pth", map_location=config["device"])
model_cnn.load_state_dict(checkpoint_cnn['model_state_dict'])
model_cnn.eval()
print(f"Loaded model from epoch {checkpoint_cnn['epoch']} with F1={checkpoint_cnn['val_f1']:.4f}")

test_loss_cnn, test_f1_cnn, test_cm_cnn = validate(model_cnn, test_loader, criterion, config["device"])
print(f"Test Loss: {test_loss_cnn:.4f}, Test F1: {test_f1_cnn:.4f}")
print("Confusion Matrix (Test):")
print(test_cm_cnn)
