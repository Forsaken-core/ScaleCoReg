import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import LabeledDataset
from model import CoModel
from src.scaling import ScaleSelector


def train_comodel(labeled, p, lr, epochs, device, batch_size=32):
    model = CoModel(p).to(device)
    loader = DataLoader(labeled, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        support, targets = [], []
        for image, label in loader:
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            prediction, embedding = model(image)
            if epoch == epochs - 1:
                support.append(embedding.detach().cpu().numpy())
                targets.append(label.cpu().numpy())
            criterion(prediction, label).backward()
            optimizer.step()

    model.fit_knn(np.concatenate(support), np.concatenate(targets))
    return model.eval()


def select_pseudo_labels(models, labeled: LabeledDataset, unlabeled: torch.Tensor, device, pool_size=100,
                         target=500, max_rounds=5000):
    labels = labeled.label.numpy()
    anchor_ids = np.arange(len(labeled))
    selectors = []
    for model in models:
        _, embedding = model.encode(labeled.image.to(device))
        selectors.append(ScaleSelector(
            embedding, labels, anchor_ids, anchor_ids, model.head_weight(),
            model.knn.n_neighbors, model.knn.p,
        ))

    pseudo_ids = [[], []]
    pseudo_labels = [[], []]
    remaining = np.arange(len(unlabeled))
    rng = np.random.default_rng()

    for _ in range(max_rounds):
        if len(remaining) == 0 or all(len(indices) == target for indices in pseudo_ids):
            break
        pool = rng.choice(remaining, size=min(pool_size, len(remaining)), replace=False)
        images = unlabeled[pool].to(device)
        (fc1, embedding1), (fc2, embedding2) = [model.encode(images) for model in models]
        selected = []

        for recipient, provider_fc, recipient_fc, embedding in (
            (1, fc1, fc2, embedding2), (0, fc2, fc1, embedding1),
        ):
            if len(pseudo_ids[recipient]) == target:
                continue
            selector = selectors[recipient]
            support_id = len(labeled) + len(pseudo_ids[recipient])
            best_score = 0.0
            for i, sample_id in enumerate(pool):
                if sample_id in selected:
                    continue
                result = selector.evaluate(embedding[i], provider_fc[i], recipient_fc[i], support_id)
                if result["score"] > best_score:
                    best_score = result["score"]
                    best = i, result

            if best_score > 0:
                i, result = best
                sample_id, label = pool[i], provider_fc[i]
                selector.append(embedding[i], label, support_id, result)
                pseudo_ids[recipient].append(sample_id)
                pseudo_labels[recipient].append(label)
                selected.append(sample_id)

        remaining = remaining[~np.isin(remaining, selected)]

    return [
        LabeledDataset(
            torch.cat((labeled.image, unlabeled[indices])),
            torch.cat((labeled.label, torch.tensor(values, dtype=labeled.label.dtype))),
        )
        for indices, values in zip(pseudo_ids, pseudo_labels)
    ]


def train(labeled, unlabeled, device, initial_epochs=200, retrain_epochs=200):
    initial = [
        train_comodel(labeled, p, lr=0.1, epochs=initial_epochs, device=device)
        for p in (2, 3)
    ]
    expanded = select_pseudo_labels(initial, labeled, unlabeled, device)
    return tuple(
        train_comodel(branch_data, p, lr=0.01, epochs=retrain_epochs, device=device)
        for p, branch_data in zip((2, 3), expanded)
    )
