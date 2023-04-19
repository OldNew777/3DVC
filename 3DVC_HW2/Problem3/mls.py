import math
import numpy as np
from sklearn import neighbors

from geometry_processing import *
from mylogger import logger
from func import *


def gaussian_fn(r, h):
    return np.exp(-r ** 2 / (2 * h ** 2))


def wendland_fn(r, h):
    return ((1 - r / h) ** 4) * (4 * r / h + 1)


def singular_fn(r, epsilon):
    return 1 / (r ** 2 + epsilon ** 2)


class VoxelGrid:
    def __init__(self, aabb):
        self.aabb = aabb
        self.v = []
        self.value = []

    def __len__(self):
        return len(self.v)

    def __str__(self):
        return f'VoxelGrid: aabb={self.aabb}, len={len(self)}, v={self.v}, value={self.value}'

    def add_point(self, v, value):
        self.v.append(v)
        self.value.append(value)

    def calculate_mls(self):
        self.v = np.array(self.v)
        self.value = np.array(self.value)
        # calculate MLS function by f(v)=value


class Voxel:
    def __init__(self, aabb: Tuple[np.ndarray, np.ndarray], voxel_size):
        super(Voxel, self).__init__()

        self.aabb = aabb
        self.voxel_size = voxel_size
        self.voxel_length = np.ceil((self.aabb[1] - self.aabb[0]) / self.voxel_size).astype(int)
        self.voxel = [VoxelGrid(
            (self.aabb[0] + self.index1toindex3(i) * self.voxel_size,
             self.aabb[0] + (self.index1toindex3(i) + 1) * self.voxel_size))
            for i in range(np.prod(self.voxel_length))]

    def __getitem__(self, v):
        index = self.v2index3(v)
        return self.voxel[self.index3toindex1(index)]

    def __str__(self):
        v_sum = 0
        for voxel_grid in self.voxel:
            v_sum += len(voxel_grid)
        return f'Voxel: aabb={self.aabb}, voxel_size={self.voxel_size}, voxel_length={self.voxel_length}, v_sum={v_sum}'

    def __repr__(self):
        return self.__str__()

    def forward(self, x: np.ndarray):
        pass

    def v2index3(self, v: np.ndarray) -> np.ndarray:
        return np.floor((v - self.aabb[0]) / self.voxel_size).astype(int)

    def index3toindex1(self, index: np.ndarray):
        return index[0] * self.voxel_length[1] * self.voxel_length[2] + \
               index[1] * self.voxel_length[2] + index[2]

    def index1toindex3(self, index) -> np.ndarray:
        return np.array([index // (self.voxel_length[1] * self.voxel_length[2]),
                         (index % (self.voxel_length[1] * self.voxel_length[2])) // self.voxel_length[2],
                         index % self.voxel_length[2]])

    def add_point(self, v, value):
        self[v].add_point(v, value)

    def calculate_mls(self):
        for voxel_grid in self.voxel:
            voxel_grid.calculate_mls()


@time_it
def mls(point_cloud: PointCloud, kdtree: neighbors.KDTree, p: np.ndarray, values: np.ndarray):
    aabb = np.min(point_cloud.v, axis=0), np.max(point_cloud.v, axis=0)
    logger.info(f'bounding box: {aabb}')
    voxel_size = np.min(aabb[1] - aabb[0]) / 30
    logger.info(f'voxel size: {voxel_size}')
    voxel = Voxel(aabb, voxel_size)
    for i in range(len(p)):
        voxel.add_point(p[i], values[i])
    logger.info(f'voxel: {voxel}')
    logger.info(f'v_sum: {len(point_cloud)}')
