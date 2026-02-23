from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class ArtifactDataset(Dataset):
    def __init__(self, files, labels, transform_0=None, transform_1=None, val_transform=None, train=True):
        self.files = files
        self.labels = labels
        self.train = train
        self.transform_0 = transform_0
        self.transform_1 = transform_1
        self.val_transform = val_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.train:
            if label == 0 and self.transform_0:
                image = self.transform_0(image)
            elif label == 1 and self.transform_1:
                image = self.transform_1(image)
        else:
            if self.val_transform:
                image = self.val_transform(image)
            else:
                image = transforms.ToTensor()(image)

        return image, torch.tensor(label, dtype=torch.long)
