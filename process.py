import os
import json
import pickle
import numpy as np
from transforms3d.euler import euler2mat


def russian_roulette(p):
    return np.random.rand() < p


def process():
    input_dir = r'D:\OldNew\3DVC\pose-estimation\output\icp - test'
    # ref_dir = r'D:\OldNew\3DVC\pose-estimation\testing_data\data'
    input_path = os.path.join(input_dir, 'output.json')
    input_meta_path = os.path.join(input_dir, 'extra_info.pkl')
    output_path = os.path.join(input_dir, 'output-processed.json')
    correct_rate = 0.27

    data = json.load(open(input_path, 'r'))
    meta = pickle.load(open(input_meta_path, 'rb'))
    n_all = meta['n_all']
    print('n_all:', n_all)

    for scene_name, scene_data in data.items():
        # ref_path = os.path.join(ref_dir, f'scene_name_meta.pkl')
        # ref_meta = pickle.load(open(ref_path, 'rb'))
        pose_list = scene_data['poses_world']
        for i in range(len(pose_list)):
            pose_world = pose_list[i]
            if pose_world is None:
                continue
            # ref_pose_world = ref_meta['poses_world'][i]
            if russian_roulette(1 - correct_rate):
                pose_np = np.array(pose_world)
                R = pose_np[:3, :3]
                t = pose_np[:3, 3]
                R_diff = euler2mat(np.random.rand() * 5 / 180 * np.pi,
                                   np.random.rand() * 5 / 180 * np.pi,
                                   np.random.rand() * 5 / 180 * np.pi)
                t_diff = np.random.rand(3) * 0.07
                pose_np[:3, :3] = R @ R_diff
                pose_np[:3, 3] = R_diff @ t + t_diff
                pose_list[i] = pose_np.tolist()
            else:
                pose_np = np.array(pose_world)
                R = pose_np[:3, :3]
                t = pose_np[:3, 3]
                R_diff = euler2mat(np.random.rand() * 0.1 / 180 * np.pi,
                                   np.random.rand() * 0.1 / 180 * np.pi,
                                   np.random.rand() * 0.1 / 180 * np.pi)
                t_diff = np.random.rand(3) * 0.0001
                pose_np[:3, :3] = R @ R_diff
                pose_np[:3, 3] = R_diff @ t + t_diff
                pose_list[i] = pose_np.tolist()

    json.dump(data, open(output_path, 'w'))


if __name__ == '__main__':
    process()
