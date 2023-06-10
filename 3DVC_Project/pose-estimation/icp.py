import math
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from sklearn import neighbors
from transforms3d.euler import euler2mat

from utils import *
from mylogger import logger
from obj_model import ObjModel
from config import *


# Initialize transformation
def icp_init(obj_model: ObjModel):
    degree_max = np.ones(3) * 2 * np.pi
    seg_max = np.ones(3, dtype=int) * config.n_seg

    # Rotate the whole model to make use of the geometric symmetry
    mask_inf = obj_model.geometric_symmetry_split == np.inf
    mask_edge = (obj_model.geometric_symmetry_split >= 3) & (obj_model.geometric_symmetry_split <= 4)
    mask_5toinf = (obj_model.geometric_symmetry_split < np.inf) & (obj_model.geometric_symmetry_split >= 5)
    degree_max[~mask_inf] /= obj_model.geometric_symmetry_split[~mask_inf]
    seg_max[mask_inf] = 1
    seg_max[mask_edge] = min(2, config.n_seg)
    seg_max[mask_5toinf] = 1

    # logger.debug(obj_model.geometric_symmetry)
    # logger.debug(obj_model.geometric_symmetry_split)
    # logger.debug(mask_inf)
    # logger.debug(mask_2)
    # logger.debug(mask_2toinf)
    # logger.debug(*seg_max)
    # exit(0)

    R_init_list = []
    for i, j, k in np.ndindex(*seg_max):
        R_init = euler2mat(i * 2 * degree_max[i] / seg_max[i],
                           j * 2 * degree_max[j] / seg_max[j],
                           k * 2 * degree_max[k] / seg_max[k])
        R_init_list.append(R_init)

    return R_init_list


# Run ICP
def icp_solve(src: np.ndarray, gt: np.ndarray, kdtree: neighbors.KDTree,
              R_init: np.ndarray = np.eye(3), t_init: np.ndarray = np.zeros(3),
              max_iterations: int = 10000, tolerance: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, float]:
    # logger.info('')
    # logger.info(f'R_init = \n{R_init}')
    # logger.info(f't_init = \n{t_init}')
    n = src.shape[0]
    R, t = R_init, t_init
    x = src
    loss = math.inf
    loss_new = 0.
    x_new = x @ R.T + t
    for i in range(max_iterations):
        # # Visualize
        # if config.visualize and config.visualize_icp_iter:
        #     visualize_point_cloud(x_new, gt)

        # Find nearest neighbors between the current source and target points parallel
        nearest_indices = kdtree.query(x_new, k=1, return_distance=False).reshape(-1)
        y = gt[nearest_indices]
        x_average = np.mean(x, axis=0)
        y_average = np.mean(y, axis=0)

        # Compute transformation matrix that minimizes the average distance between the source and target points
        H = (y - y_average).T @ (x - x_average)
        U, Sigma, VT = np.linalg.svd(H)
        R = U @ VT
        if abs(np.linalg.det(R) + 1) < 1e-2:
            VT[-1, :] *= -1
            R = U @ VT
        t = y_average - x_average @ R.T

        # Update transformation
        x_new = src @ R.T + t

        # Compute loss
        loss_new = np.linalg.norm(x_new - y).item() / n

        # Check if converged (transformation is not updated)
        if 0 < loss - loss_new < tolerance or loss_new < tolerance:
            # probably fall into local minimum
            break

        loss = loss_new

    return R, t, loss_new


def icp(src: np.ndarray, gt: np.ndarray, obj_model: ObjModel,
        max_iterations: int = 10000, tolerance: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Iterative Closest Point (ICP) algorithm.
    """

    # src.shape = (n, 3), gt.shape = (m, 3)
    assert src.shape[1] == 3
    assert gt.shape[1] == 3
    n = src.shape[0]
    m = gt.shape[0]
    # if n > m:
    #     R, t, loss = icp(gt, src, max_iterations, tolerance)
    #     R_inv = np.linalg.inv(R)
    #     return R_inv, -R_inv @ t, loss

    loss_min = math.inf
    init_list = icp_init(obj_model)
    # print(len(init_list))
    if len(init_list) == config.n_seg ** 3:
        kdtree = neighbors.KDTree(gt)
        for R_init in init_list:
            t_init = np.mean(gt, axis=0) - np.mean(src @ R_init.T, axis=0)
            R, t, loss = icp_solve(src=src, gt=gt, kdtree=kdtree, R_init=R_init, t_init=t_init,
                                   max_iterations=max_iterations, tolerance=tolerance)
            if loss < loss_min:
                R_ans, t_ans, loss_min = R, t, loss
            # print(loss / obj_model.oblique_axis)
            if loss_min / obj_model.oblique_axis < config.loss_tolerance_multi_init:
                break
    else:
        for R_init in init_list:
            t_init = np.mean(src, axis=0) - np.mean(gt @ R_init.T, axis=0)
            gt_init = gt @ R_init.T + t_init
            kdtree = neighbors.KDTree(gt_init)
            R_init_inv = np.linalg.inv(R_init)
            R, t, loss = icp_solve(src=src, gt=gt_init, kdtree=kdtree,
                                   max_iterations=max_iterations, tolerance=tolerance)
            if loss < loss_min:
                R = R_init_inv @ R
                t = R_init_inv @ (t - t_init)
                R_ans, t_ans, loss_min = R, t, loss
            # print(loss / obj_model.oblique_axis)
            if loss_min / obj_model.oblique_axis < config.loss_tolerance_multi_init:
                break

    return R_ans, t_ans, loss_min


# def test_icp():
#     max_iterations = 2000
#     n = 10000
#     x = np.random.rand(n, 3)
#     R = euler2mat(11, 62, 13)
#     t = np.array([0, 0, 0])
#     y = x @ R.T + t
#
#     R_ans, t_ans, loss = icp(x, y, max_iterations)
#     logger.debug(f"R    : \n{R}")
#     logger.debug(f"R_ans: \n{R_ans}")
#     logger.debug(f"t    : {t}")
#     logger.debug(f"t_ans: {t_ans}")
#     logger.debug(f"loss : {loss}")
#
#     y_ans = x @ R_ans.T + t_ans
#     logger.debug(f"y : \n{y}")
#     logger.debug(f"y_ans: \n{y_ans}")
#
#
# if __name__ == '__main__':
#     test_icp()
