import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from scr.dataset import ArtifactDataset
from scr.models import get_model
from utils.helpers import get_image_list_from_dir
from scr.train import train_one_epoch, validate, run_training

# Function to load configuration from YAML file
def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Load dataset file paths and labels
train_files_labels = get_image_list_from_dir("dataset/train")
val_files_labels = get_image_list_from_dir("dataset/val")
test_files_labels = get_image_list_from_dir("dataset/test")

# Separate file paths and labels
train_files, train_labels = zip(*train_files_labels)
val_files, val_labels = zip(*val_files_labels)
test_files, test_labels = zip(*test_files_labels)

# Create PyTorch datasets
train_dataset = ArtifactDataset(train_files, train_labels, train=True)
val_dataset = ArtifactDataset(val_files, val_labels, train=False)
test_dataset = ArtifactDataset(test_files, test_labels, train=False)

# Create DataLoaders for batching
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# Initialize CNN model
model_cnn = get_model("cnn", num_classes=2)
model_cnn.to(config["device"])

# Compute class weights to handle class imbalance
labels_tensor = torch.tensor(train_labels)
class_counts = torch.bincount(labels_tensor)
class_weights = 1.0 / class_counts.float()
criterion = nn.CrossEntropyLoss(weight=class_weights.to(config["device"]))

# Train the model
model_cnn, history = run_training(model_cnn, train_loader, val_loader,
                                  epochs=config["epochs"], lr=config["learning_rate"], device=config["device"],
                                  save_path="best_model_cnn.pth")

# Load the best saved model
checkpoint_cnn = torch.load("best_model_cnn.pth", map_location=config["device"])
model_cnn.load_state_dict(checkpoint_cnn['model_state_dict'])
model_cnn.eval()
print(f"Loaded model from epoch {checkpoint_cnn['epoch']} with F1={checkpoint_cnn['val_f1']:.4f}")

# Evaluate the model on the test set
test_loss_cnn, test_f1_cnn, test_cm_cnn = validate(model_cnn, test_loader, criterion, config["device"])
print(f"Test Loss: {test_loss_cnn:.4f}, Test F1: {test_f1_cnn:.4f}")
print("Confusion Matrix (Test):")
print(test_cm_cnn)
