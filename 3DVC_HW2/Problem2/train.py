import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CubeDataset
from model import Img2PcdModel
from loss import CDLoss, HDLoss
from mylogger import logger


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def main():
    # device
    device = get_device()
    torch.set_default_device(get_device())
    torch.device(get_device())

    # data path
    cube_data_path = 'cube_dataset/clean'
    output_dir = 'outputs'

    # Training hyper-parameters:
    batch_size = 8
    epoch = 100
    learning_rate = 1e-3

    # Data lists:
    # select certain numbers randomly from 0 to 99
    training_cube_list = np.random.choice(100, 50, replace=False)
    test_cube_list = np.setdiff1d(np.arange(100), training_cube_list)
    view_idx_list = np.arange(16)

    # Preperation of datasets and dataloaders:
    training_dataset = CubeDataset(cube_data_path, training_cube_list, view_idx_list, device=device)
    test_dataset = CubeDataset(cube_data_path, test_cube_list, view_idx_list, device=device)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Network:
    model = Img2PcdModel(device=device)

    # Loss:
    loss_fn = CDLoss()
    # loss_fn = HDLoss()

    # Optimizer:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    logger.info('Initialized. Start training...')

    # Training process:
    for epoch_idx in tqdm(range(epoch), ncols=80, desc='Epoch'):
        model.train()
        for batch_idx, (data_img, data_pcd) in enumerate(training_dataloader):
            # forward
            pred = model(data_img)

            # compute loss
            loss = loss_fn(pred, data_pcd)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Final evaluation process:
    model.eval()
    for batch_idx, (data_img, data_pcd, data_r) in enumerate(test_dataloader):
        # forward
        pred = model(data_img, data_r)
        # compute loss
        loss = loss_fn(pred, data_pcd)
        logger.info(f'Batch {batch_idx}: loss = {loss.item()}')

    pass


if __name__ == "__main__":
    main()
