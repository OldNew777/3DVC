import torch
from torch import nn


class Img2PcdModel(nn.Module):
    """
    A neural network of single image to 3D.
    """

    def __init__(self, device):
        super(Img2PcdModel, self).__init__()

        # init (B, 4, 256, 256)

        self.encoder = [
            # 0, x0 = (B, 32, 128, 128)
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
                torch.nn.LeakyReLU()
            ),
            # 1, x1 = (B, 32, 128, 128)
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU()
            ),
            # 2, x2 = (B, 64, 64, 64)
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU()
            ),
            # 3, x3 = (B, 128, 32, 32)
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU()
            ),
            # 4, x4 = (B, 256, 16, 16)
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU()
            ),
            # 5, x5 = (B, 512, 8, 8)
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU()
            ),
            # 6, output = (B, 512, 4, 4)
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1),
                torch.nn.LeakyReLU(),
            )
        ]

        self.encoder2decoder = [
            None,
            torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
        ]

        self.decoder0 = [
            None,
            [
                torch.nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
                torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            ],
            [
                torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
                torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            ],
            [
                torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            ],
            [
                torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            ],
            [
                torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
                torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            ],
        ]

        self.decoder1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=8192, out_features=2048),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=2048, out_features=1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=1024, out_features=256 * 3),
        )

        self.decoder_x = torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.device = device
        self.to(device)

    def forward(self, x):  # shape = (B, 3, 256, 256)
        shape = x.shape
        if shape[1] != 4:
            x_channel4 = torch.ones((shape[0], 4, shape[2], shape[3]), device=self.device)
            x_channel4[:, :shape[1], :, :] = x
            shape = x_channel4.shape
            x = x_channel4

        # encode
        x0 = self.encoder[0](x)
        x1 = self.encoder[1](x0)
        x2 = self.encoder[2](x1)
        x3 = self.encoder[3](x2)
        x4 = self.encoder[4](x3)
        x5 = self.encoder[5](x4)
        x = self.encoder[6](x5)

        # print(x0.shape)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        # print(x5.shape)
        # print(x.shape)
        # exit(0)

        x_additional = self.decoder1(x.reshape(shape[0], -1))

        # decode
        x5 = self.encoder2decoder[5](x5)
        x4 = self.encoder2decoder[4](x4)
        x3 = self.encoder2decoder[3](x3)

        x = self.decoder0[5][0](x)
        x5 = torch.nn.LeakyReLU()(x + x5)
        x5 = self.decoder0[5][1](x5)

        x = self.decoder0[4][0](x5)
        x4 = torch.nn.LeakyReLU()(x + x4)
        x4 = self.decoder0[4][1](x4)

        x = self.decoder0[3][0](x4)
        x3 = torch.nn.LeakyReLU()(x + x3)
        x3 = self.decoder0[3][1](x3)

        # predictor
        # print(x3.shape)
        x = self.decoder_x(x3)
        # print(x.shape)
        x = x.reshape(shape[0], -1, 3)
        # print(x.shape)
        # print(x_additional.shape)
        x_additional = x_additional.reshape(shape[0], -1, 3)
        # print(x_additional.shape)
        # exit(0)

        x = torch.cat((x, x_additional), dim=1)

        return x
