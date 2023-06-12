import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn.functional as F
import numpy as np
from transforms3d.euler import euler2mat
from tqdm import tqdm

from utils import *
from mylogger import logger


# class STNkD(torch.nn.Module):
#     def __init__(self, k=3):
#         super(STNkD, self).__init__()
#         self.k = k
#
#         self.relu = torch.nn.LeakyReLU()
#
#         self.conv = torch.nn.Sequential(
#             torch.nn.Conv1d(k, 64, 1),
#             torch.nn.BatchNorm1d(64),
#             self.relu,
#
#             torch.nn.Conv1d(64, 128, 1),
#             torch.nn.BatchNorm1d(128),
#             self.relu,
#
#             torch.nn.Conv1d(128, 1024, 1),
#             torch.nn.BatchNorm1d(1024),
#             self.relu,
#         )
#
#         self.fc = torch.nn.Sequential(
#             torch.nn.Linear(1024, 512),
#             torch.nn.BatchNorm1d(512),
#             self.relu,
#
#             torch.nn.Linear(512, 256),
#             torch.nn.BatchNorm1d(256),
#             self.relu,
#
#             torch.nn.Linear(256, k * k),
#         )
#
#     def forward(self, x):
#         batch_size = x.shape[0]
#         x = self.conv(x)
#         x = torch.max(x, dim=2, keepdim=True)[0]
#         x = x.view(-1, 1024)
#
#         x = self.fc(x)
#
#         iden = torch.eye(self.k).view(1, self.k * self.k).repeat(batch_size, 1)
#         if x.is_cuda:
#             iden = iden.cuda()
#         x = x + iden
#         x = x.view(-1, self.k, self.k)
#         return x


# class PointNetFeat(torch.nn.Module):
#     def __init__(self, n_points: int):
#         super(PointNetFeat, self).__init__()
#
#         self.n_points = n_points
#         self.relu = torch.nn.LeakyReLU()
#
#         self.conv1 = torch.nn.Conv1d(3, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#
#         self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
#         self.e_conv2 = torch.nn.Conv1d(64, 128, 1)
#
#         self.conv3 = torch.nn.Conv1d(128, 256, 1)
#         self.conv4 = torch.nn.Conv1d(256, 512, 1)
#         self.conv5 = torch.nn.Conv1d(512, 1024, 1)
#
#         self.average_pooling = torch.nn.AvgPool1d(self.n_points)
#         self.max_pooling = torch.nn.MaxPool1d(self.n_points)
#
#     def forward(self, x, emb):
#         x = self.relu(self.conv1(x))
#         emb = self.relu(self.e_conv1(emb))
#         pointfeat_1 = torch.cat([x, emb], 1)
#
#         x = self.relu(self.conv2(x))
#         emb = self.relu(self.e_conv2(emb))
#         pointfeat_2 = torch.cat([x, emb], 1)
#
#         x = self.relu(self.conv4(pointfeat_2))
#         x = self.conv5(x)
#
#         # x = self.relu(self.conv3(x))
#         # x = self.relu(self.conv4(x))
#         # x = self.conv5(x)
#
#         ap_x = self.average_pooling(x)
#
#         ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.n_points)
#         return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1)  # 128 + 256 + 1024


# class PoseEstimationNet(torch.nn.Module):
#     """
#     Pose estimation model for specific object
#     """
#
#     def __init__(self, n_points: int):
#         super(PoseEstimationNet, self).__init__()
#
#         self.n_points = n_points
#
#         self.cnn = None
#         self.feat = PointNetFeat(n_points)
#         self.relu = torch.nn.LeakyReLU()
#
#         # MLP
#         self.conv1_r = torch.nn.Conv1d(1408, 640, 1)
#         self.conv1_t = torch.nn.Conv1d(1408, 640, 1)
#
#         self.conv2_r = torch.nn.Conv1d(640, 256, 1)
#         self.conv2_t = torch.nn.Conv1d(640, 256, 1)
#
#         self.conv3_r = torch.nn.Conv1d(256, 128, 1)
#         self.conv3_t = torch.nn.Conv1d(256, 128, 1)
#         self.conv3_c = torch.nn.Conv1d(256, 128, 1)
#
#         self.conv4_r = torch.nn.Conv1d(128, 6, 1)  # 6D representation
#         self.conv4_t = torch.nn.Conv1d(128, 3, 1)  # translation
#
#     def forward(self, rgb, x):
#         out_img = self.cnn(rgb)
#
#         bs, di, _, _ = out_img.shape
#
#         emb = out_img.view(bs, di, -1)
#         # TODO
#
#         x = x.transpose(2, 1).contiguous()
#         ap_x = self.feat(x, emb)
#
#         rx = self.relu(self.conv1_r(ap_x))
#         tx = self.relu(self.conv1_t(ap_x))
#
#         rx = self.relu(self.conv2_r(rx))
#         tx = self.relu(self.conv2_t(tx))
#
#         rx = self.relu(self.conv3_r(rx))
#         tx = self.relu(self.conv3_t(tx))
#
#         rx = self.conv4_r(rx).view(bs, 6, self.num_points)
#         tx = self.conv4_t(tx).view(bs, 3, self.num_points)
#
#         return rx, tx


class PointNet(torch.nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()

        self.relu = torch.nn.LeakyReLU()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 1024, 1)

        self.conv6 = torch.nn.Conv1d(1024, 512, 1)
        self.conv7 = torch.nn.Conv1d(512, 256, 1)
        self.conv8 = torch.nn.Conv1d(256, 128, 1)
        self.conv9 = torch.nn.Conv1d(128, 64, 1)
        self.conv10 = torch.nn.Conv1d(64, 9, 1)

    def forward(self, x, model_points):
        n_points = x.shape[2]
        n_points_model = model_points.shape[2]
        N = n_points_model + n_points

        # (batch_size, 3, n_points)
        x = torch.concat([x, model_points], 2)  # (batch_size, 3, N)
        x = self.relu(self.conv1(x))  # (batch_size, 64, N)
        x = self.relu(self.conv2(x))  # (batch_size, 128, N)
        x = self.relu(self.conv3(x))  # (batch_size, 256, N)
        x = self.relu(self.conv4(x))  # (batch_size, 512, N)
        x = self.conv5(x)  # (batch_size, 1024, N)
        x = F.max_pool1d(x, kernel_size=N)  # (batch_size, 1024, 1)

        x = self.relu(self.conv6(x))  # (batch_size, 512, 1)
        x = self.relu(self.conv7(x))  # (batch_size, 256, 1)
        x = self.relu(self.conv8(x))  # (batch_size, 128, 1)
        x = self.relu(self.conv9(x))  # (batch_size, 64, 1)
        x = self.conv10(x)  # (batch_size, 9, 1), 6D representation + translation

        return x


def test_model():
    nn = PointNet()

    x = torch.randn(32, 3, 10)
    y = nn(x)
    print(y.shape)
