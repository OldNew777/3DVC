import trimesh
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap

from config import config
from mylogger import logger


cmap = get_cmap('rainbow', config.n_obj)
COLOR_PALETTE = np.array([cmap(i)[:3] for i in range(config.n_obj + 3)])
COLOR_PALETTE = np.array(COLOR_PALETTE * 255, dtype=np.uint8)
COLOR_PALETTE[-3] = [119, 135, 150]
COLOR_PALETTE[-2] = [176, 194, 216]
COLOR_PALETTE[-1] = [255, 255, 225]


def sample_points_even(mesh: trimesh.Trimesh, num: int) -> np.ndarray:
    sampled_points, sampled_face_index = trimesh.sample.sample_surface(mesh, num)
    return np.array(sampled_points)


class ObjModel:
    def __init__(self, csv_row):
        # read obj model's name
        self.name = csv_row['object']

        # read obj model
        self.path = os.path.join(config.data_dir, csv_row['location'])
        self.scene = trimesh.load(os.path.join(self.path, 'visual_meshes', 'visual.dae'))
        g = self.scene.geometry
        self.mesh = next(iter(g.values()))

        # read obj model's meta data (useless now)
        self.obj_class = csv_row['class']
        self.source = csv_row['source']
        self.metric = csv_row['metric']

        # read obj model's aabb
        self.aabb_min = np.array([csv_row['min_x'], csv_row['min_y'], csv_row['min_z']])
        self.aabb_max = np.array([csv_row['max_x'], csv_row['max_y'], csv_row['max_z']])
        self.width = csv_row['width']
        self.length = csv_row['length']
        self.height = csv_row['height']

        def load_symmetry(symmetry_str: str) -> np.ndarray:
            symmetry = np.ones(3)
            if symmetry_str == 'no':
                return symmetry

            symmetry_list = symmetry_str.split('|')
            for item in symmetry_list:
                # char to int by ascii
                axis = ord(item[0]) - ord('x')
                symmetry[axis] = float(item[1:])
            return symmetry

        # read obj model's geometric symmetry
        self.geometric_symmetry = csv_row['geometric_symmetry']
        self.geometric_symmetry_split = load_symmetry(self.geometric_symmetry)

        # read obj model's visual symmetry
        self.visual_symmetry = csv_row['visual_symmetry']
        self.visual_symmetry_split = load_symmetry(self.visual_symmetry)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


class ObjList:
    def __init__(self, csv_path):
        # read object.csv
        self._obj_csv_path = csv_path

        self._obj = [None for i in range(config.n_obj)]
        # Lazy load
        with open(csv_path, 'r') as csvfile:
            self._csv_data = pd.read_csv(csvfile)

    def __getitem__(self, index: int) -> ObjModel:
        if self._obj[index] is None:
            self._obj[index] = ObjModel(self._csv_data.iloc[index])
        return self._obj[index]

    def __len__(self):
        return len(self._obj)

    def __str__(self):
        text = [f'ObjList: obj_num: {len(self._obj)}, obj_csv_path: {self._obj_csv_path}']
        for obj in self._obj:
            text.append(str(obj))
        text = '\n'.join(text)
        return text

    def __repr__(self):
        return self.__str__()


def test_objList():
    training_object_csv_path = os.path.join(config.training_data_dir, 'objects_v1.csv')
    objList = ObjList(training_object_csv_path)
    logger.debug(objList)


if __name__ == '__main__':
    test_objList()
