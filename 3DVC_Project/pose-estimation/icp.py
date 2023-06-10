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


def icp(src: np.ndarray,
        gt: np.ndarray,
        max_iterations: int = 500, tolerance: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Iterative Closest Point (ICP) algorithm.
    """

    # src.shape = (n, 3), gt.shape = (m, 3)
    assert src.shape[1] == 3
    assert gt.shape[1] == 3
    n = src.shape[0]
    m = gt.shape[0]

    kdtree = neighbors.KDTree(gt)

    # Run ICP
    def icp_solve(R_init: np.ndarray, t_init: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        # logger.info('')
        # logger.info(f'R_init = \n{R_init}')
        # logger.info(f't_init = \n{t_init}')
        R_ans, t_ans = R_init, t_init
        loss = math.inf
        for i in range(max_iterations):
            # Find nearest neighbors between the current source and target points parallel
            x = src @ R_ans.T + t_ans
            nearest_indices = kdtree.query(x)[1].reshape(-1)
            y = gt[nearest_indices]
            x_average = np.mean(x, axis=0)
            y_average = np.mean(y, axis=0)

            # Compute transformation matrix that minimizes the average distance between the source and target points
            H = (y - y_average).T @ (x - x_average)
            U, Sigma, VT = np.linalg.svd(H)
            R = U @ VT
            if np.linalg.det(R) < 0:
                VT[-1, :] *= -1
                R = U @ VT
            t = y_average - x_average @ R.T

            # Update transformation
            R_ans = R @ R_ans
            t_ans = R @ t_ans + t

            # Check if converged (transformation is not updated)
            if np.allclose(R, np.eye(3), atol=tolerance) and np.allclose(t, np.zeros(3), atol=tolerance):
                # probably fall into local minimum
                break

        # Compute loss
        x = src @ R_ans.T + t_ans
        nearest_indices = kdtree.query(x)[1].reshape(-1)
        y = gt[nearest_indices]
        loss = CDLoss_np(x, y)

        return R_ans, t_ans, loss

    # Initialize transformation
    def icp_init(src: np.ndarray, gt: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        for i in range(config.n_seg):
            for j in range(config.n_seg):
                for k in range(config.n_seg):
                    R_init = euler2mat(i * 2 * np.pi / config.n_seg,
                                       j * 2 * np.pi / config.n_seg,
                                       k * 2 * np.pi / config.n_seg)
                    t_init = np.mean(gt, axis=0) - np.mean(src @ R_init.T, axis=0)
                    yield R_init, t_init

    loss_min = math.inf
    for R_init, t_init in icp_init(src, gt):
        R, t, loss = icp_solve(R_init, t_init)
        if loss < loss_min:
            R_ans, t_ans, loss_min = R, t, loss

    return R_ans, t_ans, loss_min


def test_icp():
    max_iterations = 2000
    n = 10000
    x = np.random.rand(n, 3)
    R = euler2mat(11, 62, 13)
    t = np.array([0, 0, 0])
    y = x @ R.T + t

    R_ans, t_ans, loss = icp(x, y, max_iterations)
    logger.debug(f"R    : \n{R}")
    logger.debug(f"R_ans: \n{R_ans}")
    logger.debug(f"t    : {t}")
    logger.debug(f"t_ans: {t_ans}")
    logger.debug(f"loss : {loss}")

    y_ans = x @ R_ans.T + t_ans
    logger.debug(f"y : \n{y}")
    logger.debug(f"y_ans: \n{y_ans}")


if __name__ == '__main__':
    test_icp()
