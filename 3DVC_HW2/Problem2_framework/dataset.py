import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import open3d as o3d


class CubeDataset(Dataset):
    """
    A dataset of cubes.
    """

    def __init__(self, data_path, cube_idx_list, view_idx_list, device):
        self.data_path = data_path
        self.cube_idx_list = cube_idx_list
        self.view_idx_list = view_idx_list
        self.device = device

        # prepare data list
        self.datalist = []
        for cube_idx in cube_idx_list:
            for view_idx in view_idx_list:
                self.datalist.append(os.path.join(data_path, str(cube_idx), str(view_idx)))
    
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        img_path = self.datalist[idx] + ".png"
        pcd_path = self.datalist[idx] + ".ply"

        # read image
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_np = img.transpose(2, 0, 1) / 255  # [3, 256, 256]

        # read point cloud
        pcd = o3d.io.read_point_cloud(pcd_path)
        points_np = np.array(pcd.points)  # [1024, 3]

        # convert to tensors
        img_torch = torch.from_numpy(img_np.astype(np.float32)).to(self.device)
        points_torch = torch.from_numpy(points_np.astype(np.float32)).to(self.device)

        return img_torch, points_torch
