import sys

import torch
import numpy as np
import trimesh
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
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
        self.epoch = 100
        self.learning_rate = 3e-4

        # Data lists:
        # select certain numbers randomly from 0 to 99
        np.random.seed(1234)
        self.training_cube_list = np.random.choice(100, 50, replace=False)
        self.test_cube_list = np.setdiff1d(np.arange(100), self.training_cube_list)
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

    print('Initialized. Start training...')
    # Training process:
    for epoch_idx in range(config.epoch):
        with tqdm(total=config.epoch) as t:
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

            t.set_description(f'Epoch {epoch_idx}/{config.epoch}')
            t.set_postfix(loss=loss.item())
            t.update(epoch_idx)

    # Save the model:
    torch.save(model.state_dict(), os.path.join(config.output_dir, 'model.pth'))

    # evaluate
    evaluate(model)


def evaluate(model=None):
    # Preperation of datasets and dataloaders:
    test_dataset = CubeDataset(config.cube_data_path, config.test_cube_list, config.view_idx_list, device=config.device)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, generator=config.generator)

    # Network:
    if model is None:
        model = torch.load(os.path.join(config.output_dir, 'model.pth')).to(config.device)

    # Final evaluation process:
    model.eval()
    loss_vec = []
    output_dir = os.path.join(config.output_dir, 'eval')
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


if __name__ == "__main__":
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'eval':
        evaluate()
