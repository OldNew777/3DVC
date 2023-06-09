import torch


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


class HDLoss(torch.nn.Module):
    """
    HD Loss.
    """

    def __init__(self):
        super(HDLoss, self).__init__()

    def forward(self, prediction: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        # Hausdorff distance loss
        d = torch.cdist(prediction, ground_truth)
        dimension = len(d.shape)
        d_XY = torch.max(torch.min(d, dim=dimension - 1).values, dim=dimension - 2).values
        d_YX = torch.max(torch.min(d, dim=dimension - 2).values, dim=dimension - 2).values
        return torch.max(d_XY, d_YX)
