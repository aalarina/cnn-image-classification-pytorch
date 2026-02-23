import torch
from sklearn.metrics import f1_score
from torch import optim, nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def train_one_epoch(model, loader, criterion, optimizer, device):
  model.train()
  running_loss = 0
  preds_all, labels_all = [], []

  for imgs, labels in loader:
    imgs, labels = imgs.to(device), labels.to(device)

    optimizer.zero_grad()
    outputs = model(imgs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item() * imgs.size(0)

    preds = outputs.argmax(dim=1)
    preds_all.extend(preds.cpu().numpy())
    labels_all.extend(labels.cpu().numpy())

  epoch_loss = running_loss / len(loader.dataset)
  epoch_f1 = f1_score(labels_all, preds_all, average='micro')
  cm_train = confusion_matrix(labels_all, preds_all)
  return epoch_loss, epoch_f1, cm_train

def validate(model, loader, criterion, device):
  model.eval()
  running_loss = 0
  preds_all, labels_all = [], []

  with torch.no_grad():
    for imgs, labels in loader:
      imgs, labels = imgs.to(device), labels.to(device)
      outputs = model(imgs)
      loss = criterion(outputs, labels)

      running_loss += loss.item() * imgs.size(0)
      preds = outputs.argmax(dim=1)
      preds_all.extend(preds.cpu().numpy())
      labels_all.extend(labels.cpu().numpy())

  epoch_loss = running_loss / len(loader.dataset)
  epoch_f1 = f1_score(labels_all, preds_all, average='micro')
  cm_val = confusion_matrix(labels_all, preds_all)
  return epoch_loss, epoch_f1, cm_val

def run_training(model, train_loader, val_loader, epochs=10, lr=1e-4, device, save_path="/content/best_model.pth"):
  model.to(device)

  labels_tensor = torch.tensor(train_labels)
  class_counts = torch.bincount(labels_tensor)
  class_weights = 1.0 / class_counts.float()

  criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
  optimizer = optim.Adam(model.parameters(), lr=lr)

  best_val_f1 = 0.0
  history = {'train_loss': [], 'train_f1': [], 'val_loss': [], 'val_f1': []}
  for epoch in range(epochs):
    train_loss, train_f1, train_cm = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_f1, val_cm = validate(model, val_loader, criterion, device)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_f1'].append(train_f1)
    history['val_f1'].append(val_f1)

    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f" Train Loss: {train_loss:.4f}, F1: {train_f1:.4f}")
    print(f" Val Loss:   {val_loss:.4f}, F1: {val_f1:.4f}")
    print(" Confusion Matrix (Train):")
    print(train_cm)
    print(" Confusion Matrix (Val):")
    print(val_cm)

    # Save the best model
    if val_f1 > best_val_f1:
      best_val_f1 = val_f1
      torch.save({'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'epoch': epoch,
                  'val_f1': val_f1}, save_path)
      print(" ✅ Saved new best model!")

  # Plots
  plt.figure(figsize=(12,5))
  plt.subplot(1,2,1)
  plt.plot(history['train_loss'], label='Train Loss')
  plt.plot(history['val_loss'], label='Val Loss')
  plt.legend()
  plt.title('Loss per Epoch')

  plt.subplot(1,2,2)
  plt.plot(history['train_f1'], label='Train F1')
  plt.plot(history['val_f1'], label='Val F1')
  plt.legend()
  plt.title('F1 per Epoch')
  plt.show()

  return model, history
