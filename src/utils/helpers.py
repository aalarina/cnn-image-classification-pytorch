import matplotlib.pyplot as plt
import os
from glob import glob
import numpy as np
from torch.utils.data import DataLoader
  
def get_image_list_from_dir(directory):
  """Assumes files named like image_<frame_index>_<label>.png"""
  files = sorted(glob(os.path.join(directory, '*.png')))
  items = []
  for f in files:
    name = os.path.basename(f)
    parts = name.split('_')
    # label expected as last part like '0.png' or '1.png'
    label_part = parts[-1]
    label = int(label_part.split('.')[0])
    items.append((f, label))
  return items

# Visualization helpers

def show_augmented_examples(dataset, num=8):
  # show some images from dataset (assumes transform includes ToTensor & normalization)
  loader = DataLoader(dataset, batch_size=num, shuffle=True)
  imgs, labels = next(iter(loader))
  imgs = imgs.numpy()
  imgs = imgs.transpose(0,2,3,1)

  # unnormalize
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  imgs = std * imgs + mean
  imgs = np.clip(imgs, 0, 1)

  plt.figure(figsize=(12,4))
  for i in range(min(num, imgs.shape[0])):
    plt.subplot(1, num, i+1)
    plt.imshow(imgs[i])
    plt.title(str(int(labels[i].item())))
    plt.axis('off')
  plt.show()

def count_labels(dataset):
    labels = [label for _, label in dataset]
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))
