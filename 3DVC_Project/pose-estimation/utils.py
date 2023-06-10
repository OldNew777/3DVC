from typing import Tuple
import time

import numpy as np
import cv2
from sklearn import metrics
from transforms3d.quaternions import quat2mat
import open3d as o3d

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


def CDLoss_np(x: np.ndarray, y: np.ndarray) -> float:
    """
    CD Loss.
    """
    # Chamfer Distance Loss
    d2 = metrics.euclidean_distances(x, y)
    dimension = len(d2.shape)
    d_x = np.min(d2, axis=dimension - 1)
    d_y = np.min(d2, axis=dimension - 2)
    return np.sum(d_x, axis=dimension - 2) / d2.shape[-2] + \
        np.sum(d_y, axis=dimension - 2) / d2.shape[-1]


def visualize_point_cloud(src: np.ndarray, R: np.ndarray = np.eye(3), t: np.ndarray = np.zeros(3), gt: np.ndarray = None):
    # predicted point cloud, red
    points_src = src @ R.T + t
    colors_src = np.tile([1, 0, 0], (src.shape[0], 1))

    if gt is None:
        points_gt = np.zeros((0, 3))
        colors_gt = np.zeros((0, 3))
    else:
        # gt point cloud, green
        points_gt = gt
        colors_gt = np.tile([0, 1, 0], (gt.shape[0], 1))

    # visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate([points_src, points_gt], axis=0))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate([colors_src, colors_gt], axis=0))
    o3d.visualization.draw_geometries([pcd])


VERTEX_COLORS = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
]


def get_corners():
    """Get 8 corners of a cuboid. (The order follows OrientedBoundingBox in open3d)
        (y)
        2 -------- 7
       /|         /|
      5 -------- 4 .
      | |        | |
      . 0 -------- 1 (x)
      |/         |/
      3 -------- 6
      (z)
    """
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    return corners - [0.5, 0.5, 0.5]


def get_edges(corners):
    assert len(corners) == 8
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(corners[i] == corners[j]) == 2:
                edges.append((i, j))
    assert len(edges) == 12
    return edges


def draw_projected_box3d(image, center, size, rotation, extrinsic, intrinsic, color=(0, 1, 0), thickness=1):
    corners = get_corners()  # [8, 3]
    edges = get_edges(corners)  # [12, 2]
    corners = corners * size
    corners_world = corners @ rotation.T + center
    corners_camera = corners_world @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    corners_image = corners_camera @ intrinsic.T
    uv = corners_image[:, 0:2] / corners_image[:, 2:]
    uv = uv.astype(int)

    for (i, j) in edges:
        cv2.line(
            image,
            (uv[i, 0], uv[i, 1]),
            (uv[j, 0], uv[j, 1]),
            tuple(color),
            thickness,
            cv2.LINE_AA,
        )

    for i, (u, v) in enumerate(uv):
        cv2.circle(image, (u, v), radius=1, color=VERTEX_COLORS[i], thickness=1)
    return image


def draw_all_projected_box3d(image, pose_world, box_sizes, extrinsic, intrinsic, color=(0, 1, 0), thickness=2):
    boxed_image = np.array(image)
    for i in range(len(pose_world)):
        draw_projected_box3d(boxed_image, pose_world[i][:3, 3],
                             box_sizes[i], pose_world[i][:3, :3], extrinsic, intrinsic, color, thickness)
    return boxed_image
