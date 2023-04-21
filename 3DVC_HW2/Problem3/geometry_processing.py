import os
import trimesh
import numpy as np
from typing import Tuple

from func import *
from mylogger import logger


class PointCloud:
    def __init__(self, v: np.ndarray, n: np.ndarray = None, color: np.ndarray = None):
        self.v = v
        self.n = n
        self.color = 255 * np.ones(shape=(v.shape[0], 3), dtype=int) if color is None else color
        assert len(self.v) == len(self.color)

    def __len__(self):
        return len(self.v)

    def __add__(self, other):
        return PointCloud(
            np.concatenate((self.v, other.v), axis=0),
            np.concatenate((self.n, other.n), axis=0) if self.n is not None and other.n is not None else None,
            np.concatenate((self.color, other.color), axis=0) if self.color is not None and other.color is not None else None,
        )

    @classmethod
    def from_ply(cls, filename):
        mesh = trimesh.load(filename)
        v = np.array(mesh.vertices)
        n = np.zeros_like(v)
        data = mesh.metadata['_ply_raw']['vertex']['data']
        for i in range(len(data)):
            n[i] = [data[i]['nx'], data[i]['ny'], data[i]['nz']]
        return cls(v, n)

    def export_ply(self, filename):
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        except:
            pass

        normal_exist = self.n is not None and len(self.n) == len(self.v)
        with open(filename, 'w') as file:
            file.writelines([
                'ply\n',
                'format ascii 1.0\n',
                'comment author: Xin Chen\n',
                f'element vertex {len(self.v)}\n',
                'property float x\n',
                'property float y\n',
                'property float z\n',
            ])
            if normal_exist:
                file.writelines([
                    'property float nx\n',
                    'property float ny\n',
                    'property float nz\n',
                ])
            file.writelines([
                'property uchar red\n',
                'property uchar green\n',
                'property uchar blue\n',
                'end_header\n',
            ])

            for i in range(len(self)):
                file.write(
                    f'{self.v[i][0]} {self.v[i][1]} {self.v[i][2]} '
                    + (f'{self.n[i][0]} {self.n[i][1]} {self.n[i][2]} ' if normal_exist else '') +
                    f'{self.color[i][0]} {self.color[i][1]} {self.color[i][2]}\n')

        # mesh = trimesh.Trimesh(vertices=self.v, vertex_normals=self.n if normal_exist else None, vertex_colors=self.color)
        # mesh.export(filename)
