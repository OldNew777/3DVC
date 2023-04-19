import numpy as np
from sklearn import neighbors
import torch

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
    kdtree = neighbors.KDTree(point_cloud.v)
    point_cloud, p, values = get_constraints(point_cloud, kdtree, 0.01)
    mls(point_cloud, kdtree, p, values)

