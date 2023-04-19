import numpy as np
from sklearn import neighbors
import torch
import shutil
import os

from mylogger import logger
from geometry_processing import *
from get_constraints import *
from mls import *


def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


if __name__ == '__main__':
    torch.device(get_device())
    torch.set_default_device(get_device())

    filename = 'bunny.ply'
    point_cloud = PointCloud.from_ply(filename)
    point_cloud, p, values = get_constraints(point_cloud, 0.01)

    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)
    for k in [0, 1, 2]:
        for h in [0.1, 0.01, 0.001]:
            for n_neighbors in [50, 200, 500, 1000]:
                config.k = k
                config.h = h
                config.n_neighbors = n_neighbors
                logger.info('')
                logger.info(f'{config}')
                mls(point_cloud, p, values, config)
