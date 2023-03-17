import numpy as np
from mylogger import logger
from func import *
import fractions


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


def theta(q0):
    return np.rad2deg(2 * np.arccos(q0))


def omega(q):
    return np.array([q[1], q[2], q[3]]) / np.sin(np.arccos(q[0]))


def omega_matrix(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=float)


def exponential_coordinate(q):
    _theta = theta(q[0])
    _omega = omega(q)
    exp_coord = _omega * np.deg2rad(_theta)
    return _theta, _omega, exp_coord


def exponential_coordinate2matrix(_theta, _omega):
    m_omega = omega_matrix(_omega)
    m = np.eye(3) + m_omega * np.sin(np.deg2rad(_theta)) + m_omega @ m_omega * (1 - np.cos(np.deg2rad(_theta)))
    return m


if __name__ == '__main__':
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
