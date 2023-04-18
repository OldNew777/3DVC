import numpy as np

from geometry_processing import *


def get_constraints(v: np.ndarray, n: np.ndarray):
    pass


if __name__ == '__main__':
    filename = 'bunny.ply'
    v, n = load_point_cloud(filename)
    get_constraints(v, n)
