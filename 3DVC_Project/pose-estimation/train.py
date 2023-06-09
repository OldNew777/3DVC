import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import shutil
import tqdm

import numpy as np
import torch

from mylogger import logger
from model import *
from config import *
from dataset import *


def save_state_dict(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state_dict, checkpoint_path(epoch))


def load_state_dict(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
    state_dict = torch.load(checkpoint_path(epoch))
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    assert epoch == state_dict['epoch']
    return state_dict['epoch']


def create_model(mode: str = 'train'):
    model = PoseEstimationModel()

    if mode == 'train':
        model.train()
    elif mode == 'test':
        model.eval()
    else:
        raise ValueError(f'Unknown mode {mode}')

    # Init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Load checkpoints
    start_epoch = checkpoint_epoch()
    if start_epoch is not None:
        logger.info(f'Loading checkpoint {start_epoch}')
        start_epoch = load_state_dict(model, optimizer, start_epoch)
    else:
        start_epoch = 0
        logger.info('No checkpoint found, start from scratch')

    # cuda
    if torch.cuda.is_available():
        model.cuda()
    # multi-GPUs parallel
    if len(config.device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.device_ids)

    # The learning rate scheduling is implemented with LambdaLR PyTorch scheduler.
    def lr_lambda(epoch):
        return config.lr_scheduler_gamma ** (
                epoch / config.lr_scheduler_step_size
        )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=start_epoch - 1, verbose=False
    )

    return model, optimizer, lr_scheduler, start_epoch


def train():
    model, optimizer, lr_scheduler, start_epoch = create_model('train')

    train_dataset = get_datasets('train')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )

    t_range = tqdm.tqdm(range(start_epoch, config.num_epochs))
    for epoch in t_range:
        for iteration, batch in enumerate(train_dataloader):
            rgb, depth, label, meta = batch[0].values()
            # process np raw to torch tensor
            # TODO

            # Run model forward
            # out = model()

            # Compute loss
            loss = None

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t_range.set_description(f'Epoch {epoch:05d} loss: {loss.item():.06f}')


if __name__ == "__main__":
    train()
