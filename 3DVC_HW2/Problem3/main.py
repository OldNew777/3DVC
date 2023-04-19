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

    config_t = Config()
    os.makedirs(config_t.output_dir, exist_ok=True)
    for n_neighbors in [50, 200, 500, 1000]:
        for h in [0.1, 0.01, 0.001]:
            for k in [0, 1, 2]:
                config_t.b_fn_k = k
                config_t.h_theta = h
                config_t.n_neighbors = n_neighbors
                logger.info('')
                logger.info(f'k = {k}, h = {h}, n_neighbors = {n_neighbors}')
                logger.info(f'{config_t}')
                mls(point_cloud, p, values, config_t)
