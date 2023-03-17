import numpy as np
import shutil
import os
from sklearn import neighbors, decomposition
from func import *
from mylogger import logger
import trimesh

sampled_num = 100000
sampled_num_iterative = 4000
neighbor_num = 50


@time_it
def load_mesh() -> trimesh.Trimesh:
    mesh = trimesh.load_mesh('./data/saddle.obj')
    return mesh


@time_it
def export_ply(points: np.ndarray, normals: np.ndarray, filename: str):
    mesh = trimesh.Trimesh()
    mesh.vertices = points
    if normals is not None:
        mesh.vertex_normals = normals
    mesh.export(filename)


@time_it
def sample_points_even(mesh: trimesh.Trimesh, num: int) -> np.ndarray:
    sampled_points, sampled_face_index = trimesh.sample.sample_surface_even(mesh, num)
    return np.array(sampled_points)


@time_it
def farthest_point_sampling(points: np.ndarray, num: int) -> np.ndarray:
    n, channel = points.shape
    assert channel == 3
    sampled_points = np.zeros(num, dtype=int)
    distance = np.ones(n) * np.inf

    def cal_distance(point: np.ndarray) -> np.ndarray:
        return np.linalg.norm((points - point) ** 2, axis=1)

    # choose the farthest point from the barycenter as the first point
    barycenter = (np.sum(points, axis=0) / n).reshape(1, 3)
    dist = cal_distance(barycenter)
    farthest = np.argmax(dist)

    # iteratively choose the farthest point from the sampled points
    for i in range(num):
        sampled_points[i] = farthest
        dist = cal_distance(points[farthest])
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)

    return points[sampled_points]


@time_it
def normal_estimation(points: np.ndarray, sampled_points: np.ndarray) -> np.ndarray:
    # establish a kdtree
    kdtree = neighbors.KDTree(points)

    channel = points.shape[1]
    assert channel == 3
    normals = np.zeros(sampled_points.shape)

    for i in range(sampled_points.shape[0]):
        # find the nearest neighbors of each sampled point
        neighbor_index = kdtree.query(sampled_points[i].reshape(1, -1), k=neighbor_num, return_distance=False)
        neighbor_points = points[neighbor_index].reshape(neighbor_num, 3)

        # # find n_components
        # model = decomposition.PCA(svd_solver='full')
        # points_fit = model.fit(neighbor_points)
        # contribution = np.cumsum(points_fit.explained_variance_ratio_)
        # n_components = np.argmax(contribution >= 0.95) + 1

        # # PCA to fit the plane
        # model = decomposition.PCA(n_components=n_components, svd_solver='full')
        # points_fit = model.fit(neighbor_points)
        # points_pca = points_fit.transform(neighbor_points)

        # logger.info('n_components =', n_components)
        # logger.info('points_fit =', points_fit)
        # logger.info('points_pca =', points_pca)

        # PCA to fit the plane
        # model = decomposition.PCA(n_components=2, svd_solver='full')
        model = decomposition.PCA(n_components=3, svd_solver='full')
        points_fit = model.fit(neighbor_points)

        # normal vector is the least variance direction
        # assume normal vector roughly points in the Y direction
        # normal = np.cross(points_fit.components_[0], points_fit.components_[1])
        normal = points_fit.components_[2]
        if normal[1] < 0:
            normal = -normal
        normal = normalize(normal)
        normals[i] = normal

    return normals


def test_normals(points: np.ndarray, normals: np.ndarray):
    error_num = 0
    loss_upper_bound = 0.2
    target_point = np.array([0.5, 0, 0.5]).reshape(1, -1)

    # get the normal of function y = z^2 - x^2
    # x^2 + y - z^2 = 0
    for i in range(points.shape[0]):
        point = points[i]
        normal = normals[i]
        normal_analytical = np.array([2 * point[0], 1, -2 * point[2]])
        normal_analytical = normalize(normal_analytical)
        if np.linalg.norm(normal - normal_analytical) > loss_upper_bound:
            if np.linalg.norm(point - target_point) < 0.1:
                logger.warning('Normal estimation failed:',
                               f'point={point}', f'normal={normal}', f'normal_analytical={normal_analytical}')
            error_num += 1
    logger.warning(f'Normal estimation failed {error_num}/{points.shape[0]} times')


if __name__ == '__main__':
    output_dir = os.path.relpath('./output')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 1. load and sample evenly
    mesh = load_mesh()
    points = sample_points_even(mesh, sampled_num)
    export_ply(points, None, os.path.join(output_dir, 'saddle_even.ply'))

    # 2. farthest point sampling
    sampled_points = farthest_point_sampling(points, sampled_num_iterative)

    # 3. normal estimation
    sampled_normals = normal_estimation(points, sampled_points)
    export_ply(sampled_points, sampled_normals, os.path.join(output_dir, 'saddle_generated.ply'))
    # test_normals(sampled_points, sampled_normals)

    # 4. mesh curvature estimation
    # TODO
