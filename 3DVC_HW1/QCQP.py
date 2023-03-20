import numpy as np
from mylogger import logger
from func import *


def load_data():
    data = np.load('./data/QCQP.npz')
    A = data['A']
    b = data['b']
    epsilon = data['eps']
    return A, b, epsilon


def solve(A: np.ndarray, b: np.ndarray, epsilon: float, coefficient: float) -> np.ndarray:
    # solve the QCQP problem
    m = A.shape[0]
    n = A.shape[1]

    def xTx(x: np.ndarray) -> float:
        return (x.T @ x).item()

    # 1. x^T x < epsilon, lambda = 0
    x = np.linalg.inv(A.T @ A) @ A.T @ b
    if xTx(x) < epsilon:
        return x

    # 2. x^T x = epsilon
    def h(k: float) -> np.ndarray:
        return np.linalg.inv(2 * coefficient * A.T @ A + 2 * k * np.eye(n)) @ A.T @ b * 2 * coefficient

    # solve the equation x^T x = epsilon by binary search
    k0 = 0.0
    k1 = 1.0
    x0 = h(k0)
    x1 = h(k1)
    xTx0 = xTx(x0)
    xTx1 = xTx(x1)
    while xTx1 > epsilon:
        k0 = k1
        x0 = x1
        xTx0 = xTx1
        k1 *= 2
        x1 = h(k1)
        xTx1 = xTx(x1)
    while True:
        k = (k0 + k1) / 2
        x = h(k)
        xTx_x = xTx(x)
        if abs(xTx_x - epsilon) < 1e-6:
            return x
        elif xTx_x > epsilon:
            k0 = k
        else:
            k1 = k


if __name__ == '__main__':
    A, b, epsilon = load_data()
    x = time_it(solve)(A, b, epsilon, 0.5)
    f = 0.5 * (np.linalg.norm(A @ x - b) ** 2)
    logger.info(f'x = {x}')
    logger.info(f'f(x) = {f}')
