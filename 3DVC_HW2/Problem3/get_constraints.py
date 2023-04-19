import numpy as np
from sklearn import neighbors
from tqdm import tqdm

from geometry_processing import *
from mylogger import logger
from func import *


@time_it
def get_constraints(point_cloud: PointCloud, kdtree: neighbors.KDTree, epsilon: float = 0.01)\
        -> Tuple[PointCloud, np.ndarray, np.ndarray]:

    def get_neighbor_constraint(index: int, epsilon: float) -> Tuple[np.ndarray, float]:
        ep = epsilon
        while True:
            p = point_cloud.v[index] + ep * point_cloud.n[index]
            neighbor_index = kdtree.query(p.reshape(1, -1), k=1, return_distance=False)[0][0]
            if neighbor_index == index:
                break
            ep *= 0.5
        return p, ep

    v_new = np.zeros(shape=(2 * point_cloud.v.shape[0], 3), dtype=float)
    n_new = np.zeros(shape=(2 * point_cloud.v.shape[0], 3), dtype=float)
    color_new = np.zeros(shape=(2 * point_cloud.v.shape[0], 3), dtype=int)
    p = np.zeros(shape=(3 * point_cloud.v.shape[0], 3), dtype=float)
    values = np.zeros(shape=(3 * point_cloud.v.shape[0], 1), dtype=float)
    for i in tqdm(range(len(point_cloud)), ncols=80):
        # constraint (a)
        p[3 * i], values[3 * i] = point_cloud.v[i], 0

        # constraint (b)
        p[3 * i + 1], values[3 * i + 1] = get_neighbor_constraint(i, epsilon)
        v_new[2 * i] = p[3 * i + 1]
        n_new[2 * i] = point_cloud.n[i]
        color_new[2 * i] = [0, 255, 0]

        # constraint (c)
        p[3 * i + 2], values[3 * i + 2] = get_neighbor_constraint(i, -epsilon)
        v_new[2 * i + 1] = p[3 * i + 2]
        n_new[2 * i + 1] = point_cloud.n[i]
        color_new[2 * i + 1] = [255, 0, 0]

    point_cloud += PointCloud(v_new, n_new, color_new)
    point_cloud.export_ply('bunny-constraints.ply')
    return point_cloud, p, values
