from typing import Tuple

import numpy as np
import time

from mylogger import logger


def time_it(func):
    def wrapper(*args, **kwargs):
        logger.info(f'Running {func.__name__}...')
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"Time taken by {func.__name__} is {end - start:.04f} seconds")
        return result
    return wrapper


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector.
    """
    return v / np.linalg.norm(v)


def normalize_matrix_row(m: np.ndarray) -> np.ndarray:
    """Normalize the rows of a matrix.
    """
    return m / np.linalg.norm(m, axis=1).reshape(-1, 1)


def tangent_plane(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the tangent plane of a triangle.
    Parameters
    ----------
    v0, v1, v2 : (3,) array
        The vertices of the triangle.
    """
    a = normalize(v1 - v0)
    b = v2 - v0
    normal = normalize(np.cross(a, b))
    c = normalize(np.cross(normal, a))
    return v0, a, c, normal
