import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from typing import List, Optional, Tuple
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch3d.renderer import PerspectiveCameras
import cv2

from config import config


def trivial_collate(batch):
    """
    A trivial collate function that merely returns the uncollated batch.
    """
    return batch


class LazyDataset(Dataset):
    """
    A lazy dataset
    """

    def __init__(self, dir: os.path, prefix_list: List) -> None:
        """
        Initialize the dataset for lazy load
        :param dir: The directory of the dataset
        :param prefix_list: The list of dataset entries {level}-{scene}-{variant}
        """
        self._prefix_list = prefix_list
        self._dir = dir

    def __len__(self) -> int:
        return len(self._prefix_list)

    def __getitem__(self, index: int):
        def load_image(prefix, postfix):
            filename = f'{prefix}_{postfix}.png'
            filename = os.path.join(self._dir, filename)
            image = cv2.imread(filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image

        prefix = self._prefix_list[index]
        rgb = load_image(prefix, 'color_kinect')
        depth = load_image(prefix, 'depth_kinect')
        label = load_image(prefix, 'label_kinect')
        with open(f'{prefix}_meta.pkl', 'rb') as f:
            meta = pickle.load(f)

        return rgb, depth, label, meta


def get_datasets(data_type: str = 'train') -> Dataset:
    if data_type == 'test':
        data_dir = os.path.join(config.testing_data_dir, 'data')
        filenames = os.listdir(data_dir)
        prefix_list = set()
        for filename in filenames:
            filename = filename.split('_')[0]
            prefix_list.add(filename)
        prefix_list = list(prefix_list)
    else:
        data_dir = os.path.join(config.training_data_dir, 'data')
        split_dir = os.path.join(config.training_data_dir, 'split')

        if data_type == 'train':
            with open(os.path.join(split_dir, 'train.txt'), 'r') as f:
                prefix_list = f.readlines()
            prefix_list = [prefix.strip() for prefix in prefix_list]
        elif data_type == 'validate':
            with open(os.path.join(split_dir, 'val.txt'), 'r') as f:
                prefix_list = f.readlines()
            prefix_list = [prefix.strip() for prefix in prefix_list]
        else:
            raise ValueError(f"Unknown data type {data_type}")

    dataset = LazyDataset(data_dir, prefix_list)
    return dataset
