import torch
import numpy as np
from transforms3d.euler import euler2mat
from tqdm import tqdm

from utils import *
from mylogger import logger


class PoseEstimationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
