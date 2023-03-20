from typing import Tuple

import numpy as np
from mylogger import logger
from func import *
import fractions
import time

import QCQP
import geometry_processing


def quaternion2matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to a rotation matrix.
    :param q: A quaternion of shape (4,).
    :return: A rotation matrix of shape (3, 3).
    """
    return np.array([
        [1 - 2 * q[2] ** 2 - 2 * q[3] ** 2, 2 * q[1] * q[2] - 2 * q[0] * q[3], 2 * q[1] * q[3] + 2 * q[0] * q[2]],
        [2 * q[1] * q[2] + 2 * q[0] * q[3], 1 - 2 * q[1] ** 2 - 2 * q[3] ** 2, 2 * q[2] * q[3] - 2 * q[0] * q[1]],
        [2 * q[1] * q[3] - 2 * q[0] * q[2], 2 * q[2] * q[3] + 2 * q[0] * q[1], 1 - 2 * q[1] ** 2 - 2 * q[2] ** 2]
    ])


def theta(q0: np.ndarray) -> float:
    return np.rad2deg(2 * np.arccos(q0))


def omega(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3]]) / np.sin(np.arccos(q[0]))


def omega_matrix(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=float)


def exponential_coordinate(q: np.ndarray):
    _theta = theta(q[0])
    _omega = omega(q)
    exp_coord = _omega * np.deg2rad(_theta)
    return _theta, _omega, exp_coord


def exponential_coordinate2matrix(_theta: float, _omega: np.ndarray) -> np.ndarray:
    m_omega = omega_matrix(_omega)
    m = np.eye(3) + m_omega * np.sin(np.deg2rad(_theta)) + m_omega @ m_omega * (1 - np.cos(np.deg2rad(_theta)))
    return m


def test():
    np.set_printoptions(formatter={'all': lambda x: str(fractions.Fraction(x).limit_denominator())})

    p = np.array([
        1 / np.sqrt(2),
        1 / np.sqrt(2),
        0,
        0
    ])

    q = np.array([
        1 / np.sqrt(2),
        0,
        1 / np.sqrt(2),
        0
    ])

    r = normalize((p + q) / 2)
    Mr = quaternion2matrix(r)
    logger.info(f'r = {r}')
    logger.info(f'M(r) = {Mr}')

    theta_p, omega_p, exp_coord_p = exponential_coordinate(p)
    logger.info(f'theta={theta_p}, omega={omega_p}, exp_coord={exp_coord_p}')

    theta_q, omega_q, exp_coord_q = exponential_coordinate(q)
    logger.info(f'theta={theta_q}, omega={omega_q}, exp_coord={exp_coord_q}')

    m_omega_p = omega_matrix(omega_p)
    m_p = exponential_coordinate2matrix(theta_p, omega_p)
    logger.info(f'M_omega = {m_omega_p}')
    logger.info(f'M = {m_p}')

    m_omega_q = omega_matrix(omega_q)
    m_q = exponential_coordinate2matrix(theta_q, omega_q)
    logger.info(f'M_omega = {m_omega_q}')
    logger.info(f'M = {m_q}')

    theta_p, omega_p, exp_coord_p = exponential_coordinate(-p)
    logger.info(f'theta={theta_p}, omega={omega_p}, exp_coord={exp_coord_p}')
    m_p = exponential_coordinate2matrix(theta_p, omega_p)
    logger.info(f'M = {m_p}')

    theta_q, omega_q, exp_coord_q = exponential_coordinate(-q)
    logger.info(f'theta={theta_q}, omega={omega_q}, exp_coord={exp_coord_q}')
    m_q = exponential_coordinate2matrix(theta_q, omega_q)
    logger.info(f'M = {m_q}')


def load_data() -> Tuple[int, np.ndarray, np.ndarray]:
    data = np.load('./data/teapots.npz')
    X = data['X'].T
    Y = data['Y'].T
    n = X.shape[0]
    return n, X, Y


@time_it
def solve_teapot(n: int, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # X, Y shape (n, 3)

    # init ratation matrix
    R = np.eye(3)
    R_last = np.zeros((3, 3))
    delta_omega = np.array([1., 1., 1.])
    exp_omega = exponential_coordinate2matrix(1, delta_omega)

    def f(R: np.ndarray) -> float:
        return np.linalg.norm(X @ R.T - Y, ord='fro') ** 2

    def terminated(R: np.ndarray, R_last: np.ndarray) -> bool:
        # return False
        return abs(f(R_last) - f(R)) <= 1e-6

    def g(v: np.ndarray) -> np.ndarray:
        # return omega_matrix(v)
        return R @ omega_matrix(v)

    # find a delta omega
    # print every 1 second
    epsilon = 1e-2
    iter = 0
    A = np.vectorize(g, signature='(3) -> (3, 3)')(X)
    A = A.reshape(-1, 3)
    while not terminated(R, R_last):
        R_last = R
        # A = np.vectorize(g, signature='(3) -> (3, 3)')(X)
        # A = A.reshape(-1, 3)
        # b = (X @ R.T - Y).reshape(-1, 1)
        b = (X - Y @ R).reshape(-1, 1)

        delta_omega = QCQP.solve(A, b, epsilon, 1.)
        R = R @ exponential_coordinate2matrix(1, delta_omega)

        iter += 1
        print(f'\r[Iter: {iter:05d}] f = {f(R)}', end='')

    logger.info(f'f = {f(R)}, f_last = {f(R_last)}')

    return R


if __name__ == '__main__':
    # test()

    n, X, Y = load_data()
    geometry_processing.export_ply(X, None, './output/teapots_X.ply')
    geometry_processing.export_ply(Y, None, './output/teapots_Y.ply')
    R = solve_teapot(n, X, Y)

    Y_calculated = X @ R.T
    f = np.linalg.norm(Y_calculated - Y, ord='fro')
    logger.info(f'R = {R}')
    logger.info(f'f = {f}')
    geometry_processing.export_ply(Y_calculated, None, './output/teapots_Y_calculated.ply')
