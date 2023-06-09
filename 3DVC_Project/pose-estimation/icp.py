import math

import numpy as np
from tqdm import tqdm
from sklearn import neighbors
from transforms3d.euler import euler2mat

from utils import *
from mylogger import logger


@time_it
def icp(src: np.ndarray,
        gt: np.ndarray,
        max_iterations: int = 1000, tolerance: float = 1e-5):
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
        logger.info('')
        logger.info(f'R_init = \n{R_init}')
        logger.info(f't_init = \n{t_init}')
        R_ans, t_ans = R_init, t_init
        loss = math.inf
        for i in range(max_iterations):
            # Find nearest neighbors between the current source and target points parallel
            x = src @ R_ans.T + t_ans
            nearest_indices = kdtree.query(x)[1].reshape(-1)
            y = gt[nearest_indices]
            x_average = np.mean(x, axis=0)
            y_average = np.mean(y, axis=0)

            # Compute loss
            # TODO: CD Loss?
            loss = np.linalg.norm(x - y, axis=1, ord=2).mean()

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

        return R_ans, t_ans, loss

    # Initialize transformation
    def icp_init(src: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # aabb_src = np.array([np.min(src, axis=0), np.max(src, axis=0)])
        # aabb_gt = np.array([np.min(gt, axis=0), np.max(gt, axis=0)])
        R_init = np.eye(3)
        t_init = np.mean(gt, axis=0) - np.mean(src @ R_init.T, axis=0)
        return R_init, t_init
    R_init, t_init = icp_init(src, gt)

    return icp_solve(R_init, t_init)


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
