import numpy as np
from tqdm import tqdm
from sklearn import neighbors
from transforms3d.euler import euler2mat

from utils import *
from mylogger import logger


def icp_init(src: np.ndarray, tgt: np.ndarray, kdtree) -> Tuple[np.ndarray, np.ndarray]:
    aabb_src = np.array([np.min(src, axis=0), np.max(src, axis=0)])
    aabb_tgt = np.array([np.min(tgt, axis=0), np.max(tgt, axis=0)])
    R_ans = np.eye(3)
    t_ans = np.average(aabb_tgt, axis=0) - np.average(aabb_src, axis=0)
    return R_ans, t_ans


@time_it
def icp(src: np.ndarray,
        tgt: np.ndarray,
        max_iterations: int = 100, tolerance: float = 1e-5):
    """
    Iterative Closest Point (ICP) algorithm.
    """
    assert src.shape[1] == 3
    assert tgt.shape[1] == 3
    n = src.shape[0]
    m = tgt.shape[0]

    kdtree = neighbors.KDTree(tgt[:, :3])

    # Initialize transformation
    R_ans, t_ans = icp_init(src, tgt, kdtree)

    # Run ICP
    nearest_indices = None
    for i in tqdm(range(max_iterations), desc="ICP", ncols=80):
        # Find nearest neighbors between the current source and target points parallel
        x = src @ R_ans.T + t_ans
        nearest_indices = kdtree.query(x)[1].reshape(-1)
        y = tgt[nearest_indices]
        x_average = np.average(x, axis=0)
        y_average = np.average(y, axis=0)

        loss = np.linalg.norm(x - y, axis=1, ord=2).mean()

        # Compute transformation matrix that minimizes the average distance between the source and target points
        H = y.T @ x
        U, Sigma, VT = np.linalg.svd(H)
        R = U @ VT
        if np.linalg.det(R) < 0:
            VT[-1, :] *= -1
            R = U @ VT
        t = y_average - x_average @ R.T

        R_ans = R @ R_ans
        t_ans = t + R @ t_ans

        # Check if converged (transformation is not updated)
        if np.allclose(R, np.eye(3), atol=tolerance) and np.allclose(t, np.zeros(3), atol=tolerance):
            logger.info(f"Converged at iteration {i}")
            break

    logger.debug(f'nearest_indices = {nearest_indices}')

    return R_ans, t_ans


def test_icp():
    n = 1
    x = np.random.rand(n, 3)
    R = euler2mat(11, 62, 13)
    t = np.array([1, 2, -3])
    y = x @ R.T + t

    R_ans, t_ans = icp(x, y, 1000)
    logger.debug(f"R    : \n{R}")
    logger.debug(f"R_ans: \n{R_ans}")
    logger.debug(f"t    : {t}")
    logger.debug(f"t_ans: {t_ans}")

    y_ans = x @ R_ans.T + t_ans
    logger.debug(f"y    : {y}")
    logger.debug(f"y_ans: {y_ans}")


if __name__ == '__main__':
    test_icp()
