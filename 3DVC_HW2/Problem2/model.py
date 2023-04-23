import torch
from torch import nn

from mylogger import logger


class Img2PcdModel(nn.Module):
    """
    A neural network of single image to 3D.
    """

    def __init__(self, device):
        super(Img2PcdModel, self).__init__()

        self.n_points = 256

        self.encoder0 = [
            # 0, x0
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=1),
                torch.nn.ReLU()
            ),
            # 1, x1
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU()
            ),
            # 2, x2
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU()
            ),
            # 3, x3
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU()
            ),
            # 4, x4
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU()
            ),
            # 5, x5
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU()
            ),
            # 5, output
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=2),
                torch.nn.ReLU(),
            )
        ]

        self.encoder0todecoder = [
            None,
            torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
        ]

        # TODO
        self.additional = torch.nn.Sequential(
            torch.nn.Linear(in_features=512, out_features=2048),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2048, out_features=2048),
            # TODO: use different weight_decay and add them
            torch.nn.Linear(in_features=2048, out_features=1024),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=1024, out_features=self.n_points * 3),
            torch.nn.ReLU(),
        )

        self.decoder = [
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1),
            ),
        ]

        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(in_features=512, out_features=self.n_points * 3),
            torch.nn.ReLU(),
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

        for index, l in enumerate(self.encoder0):
            logger.debug(l)
            x = l(x)
            logger.debug(x.shape)
        x = self.predictor(x)
        logger.debug(x.shape)
        x = x.reshape(shape[0], -1, 3)
        logger.debug(x.shape)
        exit(0)

        return x
