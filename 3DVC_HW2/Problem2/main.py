import json
import sys
import os
import shutil
import json

import torch
from torch.utils.data import DataLoader
import numpy as np
import trimesh
from tqdm import tqdm
import cv2


from dataset import CubeDataset
from model import Img2PcdModel
from loss import *


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def test_loss():
    torch.cuda.manual_seed(234897)
    n = 3
    b = torch.rand(size=(n, 3)) * 2 - 1
    print(f'b = {b}')
    for loss_fn in [CDLoss(), HDLoss()]:
        a = torch.rand(size=(n, 3)) * 2 - 1
        a = a.to(b.device)
        a.requires_grad = True

        print('')
        print('loss_fn =', loss_fn.__class__.__name__)
        print('init a =', a)

        loss = loss_fn(a, b)
        print('loss =', loss)

        optimizer = torch.optim.Adam([a], lr=1e-1)
        for i in range(3000):
            loss = loss_fn(a, b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('loss =', loss)
        print('a =', a)
        print('a.grad =', a.grad)
        trimesh.Trimesh(vertices=a.detach().cpu().numpy()).export(
            os.path.join('outputs', f'{loss_fn.__class__.__name__}.ply'))

    exit(0)


class Config:
    def __init__(self):
        # device
        self.device = get_device()
        torch.set_default_device(self.device)
        self.generator = torch.Generator(device=self.device)

        # hyper-parameters
        self.loss_fn = CDLoss()
        self.batch_size = 8
        self.epoch = 1000
        self.learning_rate = 1e-4
        self.save_interval = 50

        # data path
        self.dataset = 'clean'
        self.cube_data_path = f'cube_dataset/{self.dataset}'
        activation_func = Img2PcdModel.activation_func()
        activation_func_name = activation_func.__class__.__name__
        self.output_dir = f'D:/OldNew/3DVC/image2pcd/outputs-{activation_func_name}'
        if activation_func_name == torch.nn.LeakyReLU.__name__ and \
                activation_func.negative_slope != torch.nn.LeakyReLU().negative_slope:
            self.output_dir += f'-{activation_func.negative_slope}'
        self.output_dir += f'-step-{self.loss_fn.__class__.__name__}-{self.dataset}'
        os.makedirs(self.output_dir, exist_ok=True)
        self.eval_dir = os.path.join(self.output_dir, 'eval')
        if os.path.exists(self.eval_dir):
            shutil.rmtree(self.eval_dir)
        os.makedirs(self.eval_dir, exist_ok=True)
        self.model_path = os.path.join(self.output_dir, 'model')
        os.makedirs(self.model_path, exist_ok=True)

        # Data lists:
        # select certain numbers randomly from 0 to 99
        np.random.seed(1234)
        self.training_cube_list = np.random.choice(100, 80, replace=False)
        self.test_cube_list = np.setdiff1d(np.arange(100), self.training_cube_list)
        self.test_cube_list = np.random.choice(self.test_cube_list, 20, replace=False)
        self.view_idx_list = np.arange(16)


config = Config()


def get_latest_epoch_filename():
    model_filename_list = [filename for filename in os.listdir(config.model_path)
                           if filename.endswith('.pth')]
    model_epoch_idx = -1
    index = -1
    for i in range(len(model_filename_list)):
        model_filename = model_filename_list[i]
        if not model_filename.startswith('model-'):
            continue
        model_filename = model_filename.rstrip('.pth').lstrip('model-')
        try:
            epoch_idx = int(model_filename)
            if epoch_idx > model_epoch_idx:
                model_epoch_idx = epoch_idx
                index = i
        except:
            continue
    if index == -1:
        return None, 0
    return os.path.join(config.model_path, model_filename_list[index]), model_epoch_idx


def save_model(model, filename, loss_epoch_vec):
    if type(filename) is int:
        filename = os.path.join(config.model_path, f'model-{filename}.pth')
    torch.save(model.state_dict(), filename)
    json.dump(loss_epoch_vec, open(os.path.join(config.output_dir, 'training_loss.json'), 'w'), indent=4)


def load_model(filename):
    state_dict = torch.load(filename)
    model = Img2PcdModel(device=config.device)
    model.load_state_dict(state_dict)
    loss_epoch_vec = json.load(open(os.path.join(config.output_dir, 'training_loss.json'), 'r'))
    return model, loss_epoch_vec


def train():
    # Preperation of datasets and dataloaders:
    training_dataset = CubeDataset(config.cube_data_path, config.training_cube_list, config.view_idx_list,
                                   device=config.device)
    training_dataloader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True,
                                     generator=config.generator)

    # Network:
    model_filename, start_epoch = get_latest_epoch_filename()
    if model_filename is not None:
        print(f'Loading model from file "{model_filename}"', )
        model, loss_epoch_vec = load_model(model_filename)
    else:
        print('Creating new model')
        model = Img2PcdModel(device=config.device)
        loss_epoch_vec = []

    # Optimizer:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)

    # Training process:
    with tqdm(range(start_epoch, config.epoch)) as t:
        for epoch_idx in t:
            t.set_description(f'Epoch {epoch_idx}/{config.epoch}')

            model.train()
            loss_sum = torch.tensor(0.0, device=config.device)
            loss_count = torch.tensor(0, device=config.device, dtype=torch.int32)
            for batch_idx, (data_img, data_pcd) in enumerate(training_dataloader):
                # forward
                pred = model(data_img)

                # compute loss
                loss = config.loss_fn(pred, data_pcd).mean()
                loss_sum += loss * data_img.shape[0]
                loss_count += data_img.shape[0]

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            t.set_postfix(loss=loss.item())
            loss_epoch_vec.append(loss_sum.item() / loss_count.item())
            if (epoch_idx + 1) % config.save_interval == 0:
                save_model(model, epoch_idx + 1, loss_epoch_vec)

    # Save the model:
    save_model(model, config.epoch, loss_epoch_vec)

    return model


def evaluate(model=None):
    # Preperation of datasets and dataloaders:
    test_dataset = CubeDataset(config.cube_data_path, config.test_cube_list, config.view_idx_list, device=config.device)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, generator=config.generator)

    # Network:
    if model is None:
        model_filename, _ = get_latest_epoch_filename()
        if model_filename is not None:
            print(f'Loading model from file "{model_filename}"', )
            model, _ = load_model(model_filename)
        else:
            print('No model found')
            return

    # Final evaluation process:
    model.eval()
    loss_vec = []
    with torch.no_grad():
        for batch_idx, (data_img, data_pcd) in enumerate(test_dataloader):
            # forward
            pred = model(data_img)

            # compute loss
            loss = config.loss_fn(pred, data_pcd)
            loss_vec.append(loss.item())
            print(f'Batch {batch_idx}: loss = {loss.item()}')

            # save the output point cloud
            # eval output
            filename = os.path.join(config.eval_dir, f'{batch_idx}-eval.ply')
            v = pred[0].detach().cpu().numpy()
            color = np.ones_like(v) * np.array([255, 0, 0])
            trimesh.Trimesh(vertices=v, vertex_colors=color).export(filename)
            # ref
            filename = os.path.join(config.eval_dir, f'{batch_idx}.ply')
            v = data_pcd[0].detach().cpu().numpy()
            color = np.ones_like(v) * np.array([0, 255, 0])
            trimesh.Trimesh(vertices=v, vertex_colors=color).export(filename)
            # RGB image
            filename = os.path.join(config.eval_dir, f'{batch_idx}.png')
            cv2.imwrite(filename, data_img[0].detach().cpu().numpy().transpose(1, 2, 0) * 255)

    loss = np.mean(loss_vec)
    print(f'Mean loss = {loss}')

    min_loss_idx = np.argmin(loss_vec)
    print(f'Min loss = {min_loss_idx}, {loss_vec[min_loss_idx]}')

    max_loss_idx = np.argmax(loss_vec)
    print(f'Max loss = {max_loss_idx}, {loss_vec[max_loss_idx]}')

    json_dict = {
        'loss_vec': loss_vec,
        'min_loss': {
            'index': int(min_loss_idx),
            'value': float(loss_vec[min_loss_idx])
        },
        'max_loss': {
            'index': int(max_loss_idx),
            'value': float(loss_vec[max_loss_idx])
        },
        'mean_loss': float(loss),
    }
    json.dump(json_dict, open(os.path.join(config.output_dir, 'eval_loss.json'), 'w'), indent=4)


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == 'train':
        model = train()
        evaluate(model)
    elif sys.argv[1] == 'eval':
        evaluate()
