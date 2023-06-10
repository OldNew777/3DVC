import os
import torch


class Config:
    def __init__(self):
        self.algo_type = 'icp'

        self.data_dir = os.path.realpath('D:/OldNew/3DVC/pose-estimation')
        self.training_data_dir = os.path.join(self.data_dir, 'training_data')
        self.testing_data_dir = os.path.join(self.data_dir, 'testing_data')
        self.obj_model_dir = os.path.join(self.data_dir, 'model')
        self.nn_model_dir = os.path.join(self.data_dir, 'nnModel')
        self.output_dir = os.path.join(self.data_dir, 'output', self.algo_type)
        self.output_path = os.path.join(self.output_dir, 'output.json')
        self.extra_info_path = os.path.join(self.output_dir, 'extra_info.pkl')
        os.makedirs(self.nn_model_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.device_ids = [i for i in range(torch.cuda.device_count())]
        self.default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.W = 1280
        self.H = 720
        self.n_sample_points = 10000
        self.n_obj = 79
        self.n_seg = 3
        self.visualize = False
        self.visualize_icp_iter = False
        self.loss_tolerance_multi_init = 1e-5

        self.icp_max_iter = 500

        self.lr = 1e-4
        self.lr_scheduler_step_size = 1000
        self.lr_scheduler_gamma = 0.9

        self.num_epochs = 1000
        self.batch_size = 64
        self.checkpoint_interval = 50
        self.validate_interval = 100

        self.test_output_interval = 10  # value temporarily for debug


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
