import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import torch
from sklearn import metrics


def CDLoss_np(x: np.ndarray, y: np.ndarray) -> float:
    """
    CD Loss.
    """
    # Chamfer Distance Loss
    d2 = metrics.euclidean_distances(x, y)
    dimension = len(d2.shape)
    d_x = np.min(d2, axis=dimension - 1)
    d_y = np.min(d2, axis=dimension - 2)
    return np.sum(d_x, axis=dimension - 2) / d2.shape[-2] + \
        np.sum(d_y, axis=dimension - 2) / d2.shape[-1]


class CDLoss(torch.nn.Module):
    """
    CD Loss.
    """

    def __init__(self):
        super(CDLoss, self).__init__()

    def forward(self, prediction: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        # Chamfer Distance Loss
        d2 = torch.cdist(prediction, ground_truth) ** 2
        dimension = len(d2.shape)
        d_x = torch.min(d2, dim=dimension - 1).values
        d_y = torch.min(d2, dim=dimension - 2).values
        return torch.sum(d_x, dim=dimension - 2) / d2.shape[-2] + \
            torch.sum(d_y, dim=dimension - 2) / d2.shape[-1]


class HalfCDLoss(torch.nn.Module):
    """
    Half CD Loss.
    """

    def __init__(self):
        super(HalfCDLoss, self).__init__()

    def forward(self, prediction: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        # Chamfer Distance Loss
        d2 = torch.cdist(prediction, ground_truth) ** 2
        dimension = len(d2.shape)
        d_x = torch.min(d2, dim=dimension - 1).values
        return torch.sum(d_x, dim=dimension - 2) / d2.shape[-2]
