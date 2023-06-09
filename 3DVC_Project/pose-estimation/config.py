import os
import torch


class Config:
    def __init__(self):
        self.data_dir = os.path.realpath('D:/OldNew/3DVC/pose-estimation')
        self.training_data_dir = os.path.join(self.data_dir, 'training_data')
        self.testing_data_dir = os.path.join(self.data_dir, 'testing_data')
        self.obj_model_dir = os.path.join(self.data_dir, 'model')
        self.nn_model_dir = os.path.join(self.data_dir, 'nnModel')

        self.device_ids = [i for i in range(torch.cuda.device_count())]
        self.default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.W = 1280
        self.H = 720

        self.lr = 1e-4
        self.lr_scheduler_step_size = 1000
        self.lr_scheduler_gamma = 0.9

        self.num_epochs = 400
        self.batch_size = 64
        self.checkpoint_interval = 50


config = Config()


def checkpoint_epoch() -> int:
    checkpoint_dir = config.nn_model_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoints = os.listdir(checkpoint_dir)
    if len(checkpoints) == 0:
        return None

    checkpoints = [int(c.split('.')[0]) for c in checkpoints]
    return max(checkpoints)


def checkpoint_path(epoch: int) -> os.path:
    checkpoint_dir = config.nn_model_dir
    return os.path.join(checkpoint_dir, f'{epoch:05d}.pth')
