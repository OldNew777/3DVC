import numpy as np
from mylogger import logger
from func import *


def load_data():
    data = np.load('./data/QCQP.npz')
    A = data['A']
    b = data['b']
    epsilon = data['eps']
    return A, b, epsilon


@time_it
def solve(A: np.ndarray, b: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
    # solve the QCQP problem
    m = A.shape[0]
    n = A.shape[1]

    def xTx(x: np.ndarray) -> float:
        return (x.transpose() @ x).item()

    # 1. x^T x < epsilon, lambda = 0
    x = np.linalg.inv(A.transpose() @ A) @ A.transpose() @ b
    if xTx(x) < epsilon:
        return x

    # 2. x^T x = epsilon
    def h(k: float) -> np.ndarray:
        return np.linalg.inv(A.transpose() @ A + 2 * k * np.eye(n)) @ A.transpose() @ b

    # solve the equation x^T x = epsilon by binary search
    k0 = 0.0
    k1 = 1.0
    x0 = h(k0)
    x1 = h(k1)
    xTx0 = xTx(x0)
    xTx1 = xTx(x1)
    while (xTx0 - epsilon) * (xTx1 - epsilon) > 0:
        k0 = k1
        x0 = x1
        xTx0 = xTx1
        k1 *= 2
        x1 = h(k1)
        xTx1 = xTx(x1)
    while k1 - k0 > 1e-6:
        k = (k0 + k1) / 2
        x = h(k)
        xTx_x = xTx(x)
        if xTx_x == epsilon:
            return x
        elif xTx_x > epsilon:
            k0 = k
            x0 = x
            xTx0 = xTx_x
        else:
            k1 = k
            x1 = x
            xTx1 = xTx_x
    return x


if __name__ == '__main__':
    A, b, epsilon = load_data()
    x = solve(A, b, epsilon)
    logger.info(f'x = {x}')
