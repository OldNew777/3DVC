import torch
from torch import nn


class Img2PcdModel(nn.Module):
    """
    A neural network of single image to 3D.
    """

    def __init__(self, device):
        super(Img2PcdModel, self).__init__()
        # TODO: Design your network layers.
        # Example:
        #     self.linear = nn.Linear(3 * 256 * 256, 1024 * 3)
        #     self.act = nn.Sigmoid()

        self.device = device
        self.to(device)

    def forward(self, x):  # shape = (B, 3, 256, 256)
        # TODO: Design your network computation process.
        # Example:
        #     batch_size = x.shape[0]
        #     x = self.linear(x)
        #     x = self.act(x)
        #     x = x.reshape(batch_size, 1024, 3)
        #     return x

        pass
