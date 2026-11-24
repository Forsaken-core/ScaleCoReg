import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score

from model import paired_prediction


def evaluate(models, dataset, device):
    for model in models:
        model.eval()
    predicted = paired_prediction(models, dataset.image.to(device)) * 0.025
    observed = dataset.label.numpy() * 0.025
    return r2_score(observed, predicted), np.sqrt(mean_squared_error(observed, predicted))


def scaled_predictions(model, embedding, prediction, alphas):
    weight = model.head_weight().astype(np.float64)
    unit = weight / np.linalg.norm(weight)

    def scale(values, alpha):
        parallel = np.outer(values @ unit, unit)
        return parallel + alpha * (values - parallel)

    predictions = []
    for alpha in alphas:
        if alpha == 1:
            predictions.append(prediction)
        else:
            knn = clone(model.knn)
            knn.fit(scale(model.support, alpha), model.support_labels)
            predictions.append(knn.predict(scale(embedding, alpha)))
    return np.stack(predictions)


def predict_uncertainty(models, images):
    alphas = np.exp(np.log(0.5) + (np.arange(41) + 0.5) / 41 * np.log(4))
    alphas[20] = 1
    scaled, original = [], []
    for model in models:
        _, embedding, prediction = model.eval().predict(images)
        scaled.append(scaled_predictions(model, embedding, prediction, alphas) * 0.025)
        original.append(prediction.astype(np.float64) * 0.025)

    first, second = scaled
    baseline = np.mean(original, axis=0)
    disagreement = np.mean((first - second) ** 2, axis=0) / 4
    scale_deviation = np.mean(((first + second) / 2 - baseline) ** 2, axis=0)
    return (
        baseline,
        np.sqrt(disagreement),
        np.sqrt(scale_deviation),
        np.sqrt(disagreement + scale_deviation),
    )
