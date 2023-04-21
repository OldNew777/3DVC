import torch
from torch import nn


class CDLoss(nn.Module):
    """
    CD Loss.
    """

    def __init__(self):
        super(CDLoss, self).__init__()
    
    def forward(self, prediction, ground_truth):
        # TODO: Implement CD Loss
        # Example:
        #     cd_loss = torch.tensor(0, dtype=torch.float32, device=prediction.device)
        #     return cd_loss

        pass


class HDLoss(nn.Module):
    """
    HD Loss.
    """
    
    def __init__(self):
        super(HDLoss, self).__init__()
    
    def forward(self, prediction, ground_truth):
        # TODO: Implement HD Loss
        # Example:
        #     hd_loss = torch.tensor(0, dtype=torch.float32, device=prediction.device)
        #     return hd_loss

        pass
