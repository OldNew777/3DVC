from func import *
from mylogger import logger
import os
import trimesh
import numpy as np
from typing import Tuple


@time_it
def load_point_cloud(filename) -> Tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(filename)
    v = np.array(mesh.vertices)
    n = np.zeros_like(v)
    data = mesh.metadata['_ply_raw']['vertex']['data']
    for i in range(len(data)):
        n[i] = [data[i]['nx'], data[i]['ny'], data[i]['nz']]
    return v, n
