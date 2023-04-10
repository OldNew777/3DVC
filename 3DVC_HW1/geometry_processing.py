import numpy as np
import shutil
import os
from sklearn import neighbors, decomposition
from func import *
from mylogger import logger
import trimesh
import matplotlib.pyplot as plt
import tqdm

sampled_num = 100000
sampled_num_iterative = 4000
neighbor_num = 50


@time_it
def load_mesh(filename) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(filename)
    return mesh


@time_it
def export_ply(points: np.ndarray, normals: np.ndarray, filename: str):
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    mesh = trimesh.Trimesh()
    mesh.vertices = points
    if normals is not None:
        mesh.vertex_normals = normals
    if os.path.exists(filename):
        os.remove(filename)
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
    for i in tqdm.tqdm(range(num)):
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

    for i in tqdm.tqdm(range(sampled_points.shape[0])):
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


@time_it
def calculate_curvatures(mesh: trimesh.Trimesh, normals) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # calculate the curvature of each vertex
    n_faces = mesh.faces.shape[0]
    max_curvatures = np.zeros(n_faces)
    min_curvatures = np.zeros(n_faces)
    mean_curvatures = np.zeros(n_faces)
    gaussian_curvatures = np.zeros(n_faces)
    for i in tqdm.tqdm(range(n_faces)):
        face = mesh.faces[i]
        v = np.ndarray(shape=(3, 3))
        vertex_normals = np.ndarray(shape=(3, 3))
        for j in range(3):
            v[j] = mesh.vertices[face[j]]
            vertex_normals[j] = mesh.vertex_normals[face[j]] if normals is None else normals[face[j]]
        o, delta_u, delta_v, face_normal = tangent_plane(v[0], v[1], v[2])
        Df_T = np.array([delta_u, delta_v])
        A = np.zeros(shape=(6, 4))
        b = np.zeros(shape=6)
        for j in range(3):
            index_0 = (j + 1) % 3
            index_1 = (j + 0) % 3
            left_coefficient = Df_T @ (v[index_0] - v[index_1]).reshape(3, 1)
            right_coefficient = Df_T @ (vertex_normals[index_0] - vertex_normals[index_1]).reshape(3, 1)
            A[j * 2] = np.array([left_coefficient[0], left_coefficient[1], 0, 0])
            A[j * 2 + 1] = np.array([0, 0, left_coefficient[0], left_coefficient[1]])
            b[j * 2] = right_coefficient[0]
            b[j * 2 + 1] = right_coefficient[1]
        S = np.linalg.lstsq(A, b, rcond=None)[0].reshape(2, 2)
        curvatures, _ = np.linalg.eig(S)
        max_curvatures[i] = np.max(curvatures)
        min_curvatures[i] = np.min(curvatures)
        mean_curvatures[i] = np.mean(curvatures)
        gaussian_curvatures[i] = np.prod(curvatures)
    return max_curvatures, min_curvatures, mean_curvatures, gaussian_curvatures


def calculate_curvatures_and_draw_hist(obj_name: str, mesh: trimesh.Trimesh, normals: np.ndarray = None):
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    max_curvatures, min_curvatures, mean_curvatures, gaussian_curvatures = calculate_curvatures(mesh, normals)
    plt.hist(max_curvatures, bins=100)
    plt.title(f'{obj_name} max curvatures')
    plt.savefig(os.path.join(output_dir, f'{obj_name}_max_curvatures.png'))
    plt.clf()
    plt.hist(min_curvatures, bins=100)
    plt.title(f'{obj_name} min curvatures')
    plt.savefig(os.path.join(output_dir, f'{obj_name}_min_curvatures.png'))
    plt.clf()
    plt.hist(mean_curvatures, bins=100)
    plt.title(f'{obj_name} mean curvatures')
    plt.savefig(os.path.join(output_dir, f'{obj_name}_mean_curvatures.png'))
    plt.clf()
    plt.hist(gaussian_curvatures, bins=100)
    plt.title(f'{obj_name} gaussian curvatures')
    plt.savefig(os.path.join(output_dir, f'{obj_name}_gaussian_curvatures.png'))
    plt.clf()


if __name__ == '__main__':
    output_dir = os.path.relpath('./output')
    os.makedirs(output_dir, exist_ok=True)

    # 1. load and sample evenly
    mesh_saddle = load_mesh('./data/saddle.obj')
    points = sample_points_even(mesh_saddle, sampled_num)
    export_ply(points, None, os.path.join(output_dir, 'saddle_even.ply'))

    # 2. farthest point sampling
    sampled_points = farthest_point_sampling(points, sampled_num_iterative)

    # 3. normal estimation
    sampled_normals = normal_estimation(points, sampled_points)
    export_ply(sampled_points, sampled_normals, os.path.join(output_dir, 'saddle_generated.ply'))
    # test_normals(sampled_points, sampled_normals)

    # 4. mesh curvature estimation
    obj_name = 'sievert'
    mesh = load_mesh(f'./data/{obj_name}.obj')
    calculate_curvatures_and_draw_hist(obj_name, mesh)

    obj_name = 'icosphere'
    mesh = load_mesh(f'./data/{obj_name}.obj')
    calculate_curvatures_and_draw_hist(obj_name, mesh)

    # # 5. mesh curvature estimation using normal estimation
    # obj_name = 'saddle'
    # # TODO
