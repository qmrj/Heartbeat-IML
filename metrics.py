import numpy as np
from sklearn.metrics import classification_report

from typing import Any
from numpy.typing import NDArray


def get_tianchi_metric(probs: NDArray[Any], y_gt: NDArray[Any]) -> float:
    assert probs.shape[0] == y_gt.shape[0]
    target = np.zeros_like(probs)
    target[np.arange(y_gt.shape[0]), y_gt] = 1.0
    return np.sum(np.abs(probs-target))


def get_report(probs: NDArray[Any], y_gt: NDArray[Any], **kwargs):
    assert probs.shape[0] == y_gt.shape[0]
    y_pred = np.argmax(probs, axis=1)
    return classification_report(y_gt, y_pred, **kwargs)


def print_metrics(probs: NDArray[Any], y_gt: NDArray[Any]) -> None:
    print(f"Tianchi Score: {get_tianchi_metric(probs, y_gt)}\n")
    print(get_report(probs, y_gt))
