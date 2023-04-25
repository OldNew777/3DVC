import torch
from torch import nn


class Img2PcdModel(nn.Module):
    """
    A neural network of single image to 3D.
    """
    
    @classmethod
    def activation_func(cls):
        return nn.LeakyReLU()

    def __init__(self, device):
        super(Img2PcdModel, self).__init__()

        # init (B, 4, 256, 256)

        # CNN to get feature vector
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),  # (B, 64, 128, 128)
            self.activation_func(),
            nn.MaxPool2d(2, 2),  # (B, 64, 64, 64)

            nn.Conv2d(64, 128, 4, 2, 1),  # (B, 128, 32, 32)
            self.activation_func(),
            nn.MaxPool2d(2, 2),  # (B, 128, 16, 16)

            nn.Conv2d(128, 256, 4, 2, 1),  # (B, 256, 8, 8)
            self.activation_func(),
            nn.MaxPool2d(2, 2),  # (B, 256, 4, 4)
        )

        # MLP to get point cloud
        self.decoder = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024 * 3 + 256 * 2),  # (B, 1024 * 3 + 256 * 2)
            nn.Tanh(),

            nn.Linear(1024 * 3 + 256 * 2, 1024 * 3 + 256),  # (B, 1024 * 3 + 256)
            nn.Tanh(),

            nn.Linear(1024 * 3 + 256, 1024 * 3),  # (B, 1024 * 3)
        )

        self.device = device
        self.to(device)

    def forward(self, x):  # shape = (B, 3, 256, 256)
        shape = x.shape
        if shape[1] != 4:
            x_channel4 = torch.ones((shape[0], 4, shape[2], shape[3]), device=self.device)
            x_channel4[:, :shape[1], :, :] = x
            shape = x_channel4.shape
            x = x_channel4

        x = self.encoder(x)  # (B, 256, 4, 4)
        x = x.reshape(shape[0], -1)  # (B, 256 * 4 * 4)
        x = self.decoder(x)  # (B, 1024 * 3)
        x = x.reshape(shape[0], -1, 3)  # (B, 1024, 3)

        return x
