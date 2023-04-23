import sys
import os
import shutil

import torch
import numpy as np
import trimesh
from torch.utils.data import DataLoader
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

        # data path
        self.cube_data_path = 'cube_dataset/clean'
        self.output_dir = 'outputs'
        os.makedirs(self.output_dir, exist_ok=True)

        # hyper-parameters
        self.loss_fn = CDLoss()
        self.batch_size = 8
        self.epoch = 300
        self.learning_rate = 1e-2
        self.save_interval = 20

        # Data lists:
        # select certain numbers randomly from 0 to 99
        np.random.seed(1234)
        self.training_cube_list = np.random.choice(100, 30, replace=False)
        self.test_cube_list = np.setdiff1d(np.arange(100), self.training_cube_list)
        self.test_cube_list = np.random.choice(self.test_cube_list, 30, replace=False)
        self.view_idx_list = np.arange(16)


config = Config()


def train():
    # Preperation of datasets and dataloaders:
    training_dataset = CubeDataset(config.cube_data_path, config.training_cube_list, config.view_idx_list,
                                   device=config.device)
    training_dataloader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True,
                                     generator=config.generator)

    # Network:
    model = Img2PcdModel(device=config.device)

    # Optimizer:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)

    # Training process:
    with tqdm(range(config.epoch)) as t:
        for epoch_idx in t:
            t.set_description(f'Epoch {epoch_idx}/{config.epoch}')

            model.train()
            for batch_idx, (data_img, data_pcd) in enumerate(training_dataloader):
                # forward
                pred = model(data_img)

                # compute loss
                loss = config.loss_fn(pred, data_pcd).mean()

                # backward
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            t.set_postfix(loss=loss.item())
            if (epoch_idx + 1) % config.save_interval == 0:
                torch.save(model.state_dict(), os.path.join(config.output_dir, f'model.pth'))

    # Save the model:
    torch.save(model.state_dict(), os.path.join(config.output_dir, 'model.pth'))

    return model


def evaluate(model=None):
    # Preperation of datasets and dataloaders:
    test_dataset = CubeDataset(config.cube_data_path, config.test_cube_list, config.view_idx_list, device=config.device)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, generator=config.generator)

    # Network:
    if model is None:
        state_dict = torch.load(os.path.join(config.output_dir, 'model.pth'))
        model = Img2PcdModel(device=config.device)
        model.load_state_dict(state_dict)

    # Final evaluation process:
    model.eval()
    loss_vec = []
    output_dir = os.path.join(config.output_dir, 'eval')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for batch_idx, (data_img, data_pcd) in enumerate(test_dataloader):
        # forward
        pred = model(data_img)

        # compute loss
        loss = config.loss_fn(pred, data_pcd)
        loss_vec.append(loss.item())
        print(f'Batch {batch_idx}: loss = {loss.item()}')

        # save the output point cloud
        filename = os.path.join(output_dir, f'{batch_idx}.ply')
        trimesh.Trimesh(vertices=pred[0].detach().cpu().numpy()).export(filename)
        filename = os.path.join(output_dir, f'{batch_idx}.png')
        cv2.imwrite(filename, data_img[0].detach().cpu().numpy().transpose(1, 2, 0) * 255)

    loss = np.mean(loss_vec)
    print(f'Mean loss = {loss}')

    min_loss_idx = np.argmin(loss_vec)
    print(f'Min loss = {min_loss_idx}, {loss_vec[min_loss_idx]}')

    max_loss_idx = np.argmax(loss_vec)
    print(f'Max loss = {max_loss_idx}, {loss_vec[max_loss_idx]}')


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == 'train':
        model = train()
        evaluate(model)
    elif sys.argv[1] == 'eval':
        evaluate()
