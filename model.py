import numpy as np
import torch
from sklearn.neighbors import KNeighborsRegressor
from torch import nn

from src.resnet import ResNet18


class CoModel(nn.Module):
    def __init__(self, p, k=3):
        super().__init__()
        self.backbone = ResNet18()
        self.knn = KNeighborsRegressor(n_neighbors=k, p=p)

    def forward(self, images):
        return self.backbone(images)

    @torch.no_grad()
    def encode(self, images):
        fc, embedding = self(images)
        return fc.cpu().numpy(), embedding.cpu().numpy()

    def fit_knn(self, embedding, labels):
        self.support = embedding
        self.support_labels = labels
        self.knn.fit(embedding, labels)

    def predict(self, images):
        fc, embedding = self.encode(images)
        return fc, embedding, self.knn.predict(embedding)

    def head_weight(self):
        return self.backbone.fc.weight.detach().cpu().numpy().ravel()


def paired_prediction(models, images):
    return np.mean([model.predict(images)[2] for model in models], axis=0)
