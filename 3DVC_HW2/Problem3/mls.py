import math
import os.path

import numpy as np
from tqdm import tqdm
from sklearn import neighbors
import mcubes

from geometry_processing import *
from mylogger import logger
from func import *


def gaussian_fn(r, h):
    return np.exp(-r ** 2 / (2 * h ** 2))


def wendland_fn(r, h):
    return ((1 - r / h) ** 4) * (4 * r / h + 1)


def singular_fn(r, epsilon: float):
    return 1 / (r ** 2 + epsilon ** 2)


def get_theta_fn():
    return gaussian_fn


def a_num(k: int):
    return int((3 ** (k + 1) - 1) / 2)


def b_fn_0(x: np.ndarray) -> np.ndarray:
    l = x.shape[:-1]
    if l == ():
        l = (1,)
    return np.ones(shape=l)


def b_fn_1(x: np.ndarray) -> np.ndarray:
    return np.array([1, x[0], x[1], x[2]])


def b_fn_2(x: np.ndarray) -> np.ndarray:
    return np.array([1, x[0], x[1], x[2],
                     x[0] ** 2, x[0] * x[1], x[0] * x[2],
                     x[1] ** 2, x[1] * x[2], x[2] ** 2])


b_fn_dict = {
    0: b_fn_0,
    1: b_fn_1,
    2: b_fn_2,
}


class Config:
    def __init__(self):
        self.b_fn_k = 1
        self.h_theta = 0.01
        self.n_neighbors = 500
        self.voxel_length_basic = 30

        self.output_dir = 'outputs'
        self.theta_fn = None
        self.b_fn = None

    def copy(self):
        ans = Config()
        ans.b_fn_k = self.b_fn_k
        ans.h_theta = self.h_theta
        ans.n_neighbors = self.n_neighbors
        return ans

    def generate_fn(self):
        self.theta_fn = get_theta_fn()
        self.b_fn = b_fn_dict[self.b_fn_k]

    def __str__(self):
        return f'Config(b_fn_k={self.b_fn_k}, h_theta={self.h_theta}, n_neighbors={self.n_neighbors})'

    def __repr__(self):
        return str(self)


config = Config()


# def cal_mls(x: np.ndarray, v_neighbors: np.ndarray, value_neighbors: np.ndarray) -> \
#         Tuple[np.ndarray, np.ndarray]:
#     # logger.debug(f'x={x}')
#     theta = np.sqrt(config.theta_fn(np.linalg.norm(v_neighbors - x.reshape(1, 3), axis=1, ord=2), config.h_theta))
#     # logger.debug(f'theta={theta}')
#     B_x = theta.reshape(-1, 1) * np.array([config.b_fn(v) for v in v_neighbors])
#     # logger.debug(f'theta.reshape(-1, 1)={theta.reshape(-1, 1)}')
#     # logger.debug(f'B_x={B_x}')
#     d = value_neighbors * theta
#     # logger.debug(f'd={d}')
#     a = np.linalg.lstsq(B_x, d, rcond=None)[0].reshape(-1)
#     # logger.debug(f'a={a}')
#     f_value = np.dot(config.b_fn(x), a)
#     # logger.debug(f'f_value={f_value}')
#     return a, f_value


# class VoxelGrid:
#     def __init__(self, aabb):
#         self.aabb = aabb
#
#         self.a = np.zeros(shape=(8, a_num(config.b_fn_k)))
#         self.f_value = np.zeros(shape=8)
#
#     def __str__(self):
#         return f'VoxelGrid: aabb={self.aabb}'
#
#     def calculate_mls(self, p: np.ndarray, values: np.ndarray):
#         kdtree = neighbors.KDTree(p)
#
#         # calculate MLS function by f(v)=value
#
#         count = 0
#         for i in range(2):
#             for j in range(2):
#                 for k in range(2):
#                     x = np.array([self.aabb[i][0], self.aabb[j][1], self.aabb[k][2]])
#                     index_neighbors = kdtree.query(x.reshape(1, 3), k=config.n_neighbors, return_distance=False)[0]
#                     a, f_value = cal_mls(x=x, v_neighbors=p[index_neighbors], value_neighbors=values[index_neighbors])
#                     self.a[count] = a
#                     self.f_value[count] = f_value
#                     count += 1
#
#     def f(self, x: np.ndarray):
#         # use multi-linear interpolation to calculate f(x)
#         k_axis = (x - self.aabb[0]) / (self.aabb[1] - self.aabb[0])
#         k_axis = np.clip(k_axis, 0, 1)
#         f_value = self.f_value.copy()
#         for axis in range(3):
#             for i in range(0, 2 ** (3 - axis), 2 ** (axis + 1)):
#                 index_offset = 2 ** axis
#                 f_value[i] = (1 - k_axis[axis]) * f_value[i] + k_axis[axis] * f_value[i + index_offset]
#         return f_value[0]


class Voxel:
    def __init__(self, aabb: np.ndarray, voxel_size):
        super(Voxel, self).__init__()

        self.aabb = aabb
        self.voxel_length = np.ceil((self.aabb[1] - self.aabb[0]) / voxel_size).astype(int)
        self.voxel_size = (self.aabb[1] - self.aabb[0]) / self.voxel_length
        # self.voxel = [VoxelGrid(
        #     (self.aabb[0] + self.index1toindex3(i) * self.voxel_size,
        #      self.aabb[0] + (self.index1toindex3(i) + 1) * self.voxel_size))
        #     for i in range(np.prod(self.voxel_length))]

    # def __getitem__(self, v):
    #     index = self.index3toindex1(self.v2index3(v))
    #     return self.voxel[index]

    def __str__(self):
        return f'Voxel: aabb={self.aabb}, voxel_size={self.voxel_size}, voxel_length={self.voxel_length}'

    def __repr__(self):
        return self.__str__()

    def v2index3(self, v: np.ndarray) -> np.ndarray:
        index = (v - self.aabb[0]) / self.voxel_size
        index_int = np.floor(index).astype(int)
        index_int[index_int == index] -= 1
        return index_int

    def index3toindex1(self, index: np.ndarray):
        return index[0] * self.voxel_length[1] * self.voxel_length[2] + \
               index[1] * self.voxel_length[2] + index[2]

    def index1toindex3(self, index) -> np.ndarray:
        return np.array([index // (self.voxel_length[1] * self.voxel_length[2]),
                         (index % (self.voxel_length[1] * self.voxel_length[2])) // self.voxel_length[2],
                         index % self.voxel_length[2]])

    # def calculate_mls(self):
        # for voxel_grid in self.voxel:
        #     voxel_grid.calculate_mls()

    # def f(self, x: np.ndarray):
    #     return self[x].f(x)


@time_it
def mls(point_cloud: PointCloud, p: np.ndarray, values: np.ndarray,
        config_t: Config = Config()):
    global config
    config = config_t.copy()
    config.generate_fn()

    aabb = np.array([np.min(point_cloud.v, axis=0), np.max(point_cloud.v, axis=0)])
    logger.info(f'bounding box: {aabb}')
    voxel_size = np.min(aabb[1] - aabb[0]) / config.voxel_length_basic
    logger.info(f'voxel size: {voxel_size}')
    voxel = Voxel(aabb, voxel_size)
    logger.info(f'voxel: {voxel}')
    logger.info(f'v_sum: {len(point_cloud)}')

    # voxel.calculate_mls()
    # logger.debug(voxel.f(p[2342]))

    kdtree = neighbors.KDTree(p)

    def cal_mls(x_: float, y_: float, z_: float):
        x = np.array([x_, y_, z_])
        index_neighbors = kdtree.query(x.reshape(1, 3), k=config.n_neighbors, return_distance=False)[0]
        v_neighbors = p[index_neighbors]
        value_neighbors = values[index_neighbors]

        # logger.debug(f'x={x}')
        theta = np.sqrt(config.theta_fn(np.linalg.norm(v_neighbors - x.reshape(1, 3), axis=1, ord=2), config.h_theta))
        # logger.debug(f'theta={theta}')
        B_x = theta.reshape(-1, 1) * np.array([config.b_fn(v) for v in v_neighbors])
        # logger.debug(f'theta.reshape(-1, 1)={theta.reshape(-1, 1)}')
        # logger.debug(f'B_x={B_x}')
        d = value_neighbors * theta
        # logger.debug(f'd={d}')
        a = np.linalg.lstsq(B_x, d, rcond=None)[0].reshape(-1)
        # logger.debug(f'a={a}')
        f_value = np.dot(config.b_fn(x), a)
        # logger.debug(f'f_value={f_value}')
        return f_value

    # vertices, triangles = mcubes.marching_cubes_func(
    #     tuple(aabb[0]), tuple(aabb[1]),
    #     voxel.voxel_length[0], voxel.voxel_length[1], voxel.voxel_length[2],
    #     cal_mls, 16)

    f_values = np.zeros(tuple(voxel.voxel_length + 1))
    size = voxel.voxel_length + 1
    x_grid = np.linspace(aabb[0][0], aabb[1][0], size[0])
    y_grid = np.linspace(aabb[0][1], aabb[1][1], size[1])
    z_grid = np.linspace(aabb[0][2], aabb[1][2], size[2])

    for index1 in tqdm(range(np.cumprod(size, axis=0)[-1])):
        index3 = voxel.index1toindex3(index1)
        i, j, k = index3[0], index3[1], index3[2]
        f_values[i, j, k] = cal_mls(x_grid[i], y_grid[j], z_grid[k])

    vertices, triangles = mcubes.marching_cubes(-f_values, 0)
    filename = f'bunny-mls-k={config.b_fn_k}-h={config.h_theta}-n_neighbors={config.n_neighbors}.obj'
    mcubes.export_obj(vertices, triangles, os.path.join(config.output_dir, filename))
