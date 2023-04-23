import torch
import numpy as np
import trimesh
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

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
        trimesh.Trimesh(vertices=a.detach().cpu().numpy()).export(os.path.join('outputs', f'{loss_fn.__class__.__name__}.ply'))

    exit(0)


def main():
    # device
    device = get_device()
    torch.set_default_device(device)

    # data path
    cube_data_path = 'cube_dataset/clean'
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Training hyper-parameters:
    batch_size = 8
    epoch = 100
    learning_rate = 1e-3

    # test_loss()

    # Data lists:
    # select certain numbers randomly from 0 to 99
    training_cube_list = np.random.choice(100, 50, replace=False)
    test_cube_list = np.setdiff1d(np.arange(100), training_cube_list)
    view_idx_list = np.arange(16)

    # Preperation of datasets and dataloaders:
    training_dataset = CubeDataset(cube_data_path, training_cube_list, view_idx_list, device=device)
    test_dataset = CubeDataset(cube_data_path, test_cube_list, view_idx_list, device=device)
    generator = torch.Generator(device=device)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, generator=generator)

    # Network:
    model = Img2PcdModel(device=device)

    # Loss:
    loss_fn = CDLoss()
    # loss_fn = HDLoss()

    # Optimizer:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    print('Initialized. Start training...')

    # Training process:
    for epoch_idx in tqdm(range(epoch), ncols=80, desc='Epoch'):
        model.train()
        for batch_idx, (data_img, data_pcd) in enumerate(training_dataloader):
            data_img = torch.ones(size=(batch_size, 3, 192, 256), device=device)

            # forward
            pred = model(data_img)

            # compute loss
            loss = loss_fn(pred, data_pcd)

            # backward
            optimizer.zero_grad()
            loss.sum().backward(retain_graph=True)
            optimizer.step()

    # Save the model:
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))

    # Final evaluation process:
    model.eval()
    for batch_idx, (data_img, data_pcd, data_r) in enumerate(test_dataloader):
        # forward
        pred = model(data_img, data_r)
        # compute loss
        loss = loss_fn(pred, data_pcd)
        print(f'Batch {batch_idx}: loss = {loss.item()}')

    pass


if __name__ == "__main__":
    main()
