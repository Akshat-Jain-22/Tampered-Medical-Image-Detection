import os
import torch
import numpy as np
import pydicom
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from app.config.settings import DATASET_PATH, folders, BATCH_SIZE
from app.services.preprocess import train_transform, val_transform

data, labels = [], []

for folder, label in folders.items():
    folder_path = os.path.join(DATASET_PATH, folder)

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)

            if not os.path.isfile(full_path):
                continue

            data.append(full_path)
            labels.append(label)

# SPLIT
train_data, temp_data, train_labels, temp_labels = train_test_split(
    data, labels, test_size=0.3, stratify=labels, random_state=42
)

val_data, test_data, val_labels, test_labels = train_test_split(
    temp_data, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

# DATASET
class DicomDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        try:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array.astype(np.float32)

            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).astype(np.uint8)

            if len(img.shape) == 2:
                img = np.stack([img]*3, axis=-1)

        except Exception as e:
            print(f"Error reading: {path}")
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            img = self.transform(image=img)["image"]

        label = torch.tensor(self.labels[idx]).long()
        return img, label, path


# LOADERS
train_loader = DataLoader(DicomDataset(train_data, train_labels, train_transform),
                          batch_size=BATCH_SIZE, shuffle=True)

val_loader = DataLoader(DicomDataset(val_data, val_labels, val_transform),
                        batch_size=BATCH_SIZE)

test_loader = DataLoader(DicomDataset(test_data, test_labels, val_transform),
                         batch_size=BATCH_SIZE)