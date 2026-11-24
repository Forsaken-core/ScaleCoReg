import numpy as np
import torch
from torch.utils.data import Dataset


class LabeledDataset(Dataset):
    def __init__(self, image, label):
        self.image = torch.as_tensor(image, dtype=torch.float32)
        self.label = torch.as_tensor(label, dtype=torch.float32)

    @classmethod
    def from_npz(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(data["image"], data["label"])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.image[index], self.label[index]
