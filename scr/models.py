import torch.nn as nn
from torchvision import models

class ArtifactCNN(nn.Module):
  def __init__(self, num_classes=2):
    super(ArtifactCNN, self).__init__()
    self.feature_extractor = nn.Sequential(
        # First conv layer: 3 input channels (RGB), 16 output channels
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ELU(),
        nn.MaxPool2d(2), # 112
        nn.Dropout(0.2),

        # Second conv layer: 16 -> 32 channels
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ELU(),
        nn.MaxPool2d(2), # 56
        nn.Dropout(0.3),

        # Third conv layer: 32 -> 64 channels
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ELU(),
        nn.MaxPool2d(2), # 28
        nn.Dropout(0.4),

        # Optional: fourth conv layer
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ELU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.5),

        nn.AdaptiveAvgPool2d((7,7)),
        nn.Flatten(),
    )

    self.classifier = nn.Sequential(
            nn.Linear(128*7*7, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

  def forward(self, x):
    x = self.feature_extractor(x)
    x = self.classifier(x)
    return x

def get_resnet18(num_classes=2, pretrained=False):
    model = models.resnet18(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
