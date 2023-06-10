import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import shutil
import tqdm
import json

import numpy as np
import torch

from mylogger import logger
from model import *
from config import *
from dataset import *
from obj_model import *
from icp import icp
from eval import *


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
    validate_dataset = get_datasets('validate')
    validate_dataloader = torch.utils.data.DataLoader(
        validate_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )

    t_range = tqdm.tqdm(range(start_epoch, config.num_epochs))
    for epoch in t_range:
        for iteration, batch in enumerate(train_dataloader):
            rgb, depth, label, meta, prefix = batch[0].values()
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

        # Adjust the learning rate
        lr_scheduler.step()

        # Checkpoint
        if epoch % config.checkpoint_interval == 0 and epoch > 0:
            logger.info(f'Saving checkpoint {epoch}')
            save_state_dict(model, optimizer, epoch)

        # Validate
        if epoch % config.validate_interval == 0 and epoch > 0:
            for iteration, batch in enumerate(validate_dataloader):
                rgb, depth, label, meta, prefix = batch[0].values()
                # process np raw to torch tensor
                # TODO

                # Run model forward
                # out = model()

                # Compute loss
                loss = None


def load_meta(obj_model_list: ObjList, rgb: np.ndarray, depth: np.ndarray, label: np.ndarray, meta: dict):
    n_obj = len(meta['object_ids'])
    extrinsic = meta['extrinsic']
    intrinsic = meta['intrinsic']
    H = rgb.shape[0]
    W = rgb.shape[1]

    z = depth
    v, u = np.indices(z.shape)
    uv1 = np.stack([u + 0.5, v + 0.5, np.ones_like(z)], axis=-1)
    camera_space = uv1 @ np.linalg.inv(intrinsic).T * z[..., None]  # [H, W, 3]
    extinstic_R = extrinsic[:3, :3]
    extinstic_t = extrinsic[:3, 3]
    world_space = (camera_space - extinstic_t.T) @ np.linalg.inv(extinstic_R).T

    for obj_id in meta['object_ids']:
        obj_model = obj_model_list[obj_id]
        scales = meta['scales'][obj_id].reshape(1, 3)
        model_coord = sample_points_even(obj_model.mesh, config.n_sample_points) * scales

        pose_world = meta['poses_world'][obj_id]
        box_sizes = meta['extents'][obj_id] * scales
        # logger.debug(f'camera_space.shape: {camera_space.shape}')
        # logger.debug(f'extrinsic.shape: {extrinsic.shape}')
        # exit(0)

        # find the position where label == index
        mask = np.abs(label - obj_id) < 1e-1
        # generate points from depth and mask
        world_coord = world_space[mask]

        yield world_coord, model_coord, pose_world, box_sizes


@time_it
def test(algo_type: str = 'icp', nn_info: Tuple = None):
    test_dataset = get_datasets('train')
    obj_csv_path = os.path.join(config.testing_data_dir, 'objects_v1.csv')
    obj_model_list = ObjList(obj_csv_path)
    output = {}

    if algo_type == 'icp':
        n_all = 0
        n_correct = 0
        with tqdm(range(len(test_dataset)), desc='ICP', ncols=80) as pbar:
            for i in pbar:
                rgb, depth, label, meta, prefix = test_dataset[i]
                output[prefix] = {}
                pose_world_predict_list = [None for _ in range(config.n_obj)]

                for obj_index, (world_coord, model_coord, pose_world, box_sizes) in \
                        enumerate(load_meta(obj_model_list, rgb, depth, label, meta)):
                    # try to match part of the model to the whole
                    R, t, loss = icp(world_coord, model_coord, config.icp_max_iter)
                    R_inv = np.linalg.inv(R)
                    R, t = R_inv, -R_inv @ t

                    pose_world_pred = np.eye(4)
                    pose_world_pred[:3, :3] = R
                    pose_world_pred[:3, 3] = t
                    obj_id = meta['object_ids'][obj_index]
                    pose_world_predict_list[obj_id] = list(pose_world_pred)

                    # evaluate whether the prediction is correct
                    r_diff, t_diff = eval(pose_world_pred, pose_world, obj_model_list[obj_id].geometric_symmetry)
                    logger.info(f'------------------{prefix}------------------')
                    logger.info(f'obj_id = {obj_id}, obj_name = {obj_model_list[obj_id].name}')
                    logger.info(f'loss = {loss:.06f}')
                    logger.info(f'pose_world_pred =\n{pose_world_pred}')
                    logger.info(f'pose_world =\n{pose_world} degree')
                    logger.info(f'geometric_symmetry = {obj_model_list[obj_id].geometric_symmetry}')
                    logger.info(f"r_diff = {r_diff:.03f} degree, t_diff = {t_diff:.03f} cm")
                    # exit(0)
                    match = judge(r_diff, t_diff)
                    n_all += 1
                    n_correct += match
                    correct_rate = n_correct / n_all
                    pbar.set_postfix_str(f'correct ({n_correct}/{n_all}, {correct_rate:.06f})')

                    # visualize
                    if config.visualize:
                        R_ref = pose_world[:3, :3]
                        t_ref = pose_world[:3, 3]
                        visualize_point_cloud(world_coord, model_coord @ R.T + t, model_coord @ R_ref.T + t_ref)

                output[prefix]['poses_world'] = pose_world_predict_list

                if i % config.test_output_interval == 0 and i > 0:
                    with open(os.path.join(config.output_dir, 'output.json'), 'w') as f:
                        json.dump(output, f, indent=4)

        with open(os.path.join(config.output_dir, 'output.json'), 'w') as f:
            json.dump(output, f, indent=4)

    elif algo_type == 'nn':
        if nn_info is None:
            model, optimizer, lr_scheduler, start_epoch = create_model('test')
        else:
            model, optimizer, lr_scheduler, start_epoch = nn_info

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda batch: batch,
        )

        with torch.no_grad():
            # tqdm
            for iteration, batch in enumerate(tqdm.tqdm(test_dataloader, desc='NN', ncols=80)):
                rgb, depth, label, meta, prefix = batch[0].values()
                # process np raw to torch tensor
                # TODO

                # Run model forward
                # out = model()

                # Compute loss
                loss = None
    else:
        raise ValueError(f'Unknown algo_type {algo_type}')


def main():
    if config.algo_type == 'nn':
        train()
        test('nn')
    elif config.algo_type == 'icp':
        test('icp')


if __name__ == "__main__":
    main()
