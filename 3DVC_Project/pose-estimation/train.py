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


def save_state_dict(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, obj_id: int):
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state_dict, checkpoint_path(epoch, obj_id))


def load_state_dict(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, obj_id: int):
    state_dict = torch.load(checkpoint_path(epoch, obj_id))
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    assert epoch == state_dict['epoch']
    return state_dict['epoch']


def create_model(mode: str = 'train'):
    model = PointNet()

    if mode != 'train' and mode != 'test':
        raise ValueError(f'Unknown mode {mode}')

    obj_nn_models = []

    # Load checkpoints
    start_epoch = checkpoint_epoch()
    load_from_checkpoint = start_epoch is not None
    if load_from_checkpoint:
        logger.info(f'Loading checkpoint {start_epoch}')
    else:
        logger.info('No checkpoint found, start from scratch')
        start_epoch = 0

    for obj_id in range(config.n_obj):
        # Init
        model = PointNet()
        if mode == 'train':
            model.train()
        elif mode == 'test':
            model.eval()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        if load_from_checkpoint:
            load_state_dict(model, optimizer, start_epoch, obj_id)

        # multi-GPUs parallel
        if config.multi_gpu:
            model = torch.nn.DataParallel(model, device_ids=config.device_ids)
        # cuda
        if torch.cuda.is_available():
            model.cuda(device=config.default_device)

        # The learning rate scheduling is implemented with LambdaLR PyTorch scheduler.
        def lr_lambda(epoch):
            return config.lr_scheduler_gamma ** (
                    epoch / config.lr_scheduler_step_size
            )

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, last_epoch=start_epoch - 1, verbose=False
        )

        obj_nn_models.append((model, optimizer, lr_scheduler))

    return start_epoch, obj_nn_models


def train():
    start_epoch, obj_nn_models = create_model('train')
    n_multi_gpu = len(config.device_ids) if config.multi_gpu else 1

    train_dataset = get_datasets('train')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size * n_multi_gpu,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )
    validate_dataset = get_datasets('validate')
    validate_dataloader = torch.utils.data.DataLoader(
        validate_dataset,
        batch_size=config.batch_size * n_multi_gpu,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )
    obj_csv_path = os.path.join(config.training_data_dir, 'objects_v1.csv')
    obj_model_list = ObjList(obj_csv_path)

    t_range = tqdm(range(start_epoch, config.num_epochs), ncols=160)
    for epoch in t_range:
        loss_batch = torch.zeros(1)
        loss_n = 0
        for iteration, batch in enumerate(train_dataloader):
            batch_size = len(batch)
            for i in range(batch_size):
                rgb, depth, label, meta, prefix = batch[i]

                for obj_index, (world_coord, model_coord, pose_world, box_sizes) in \
                        enumerate(zip(*load_data(obj_model_list, rgb, depth, label, meta))):
                    # skip if label map is wrong
                    if world_coord.shape[0] == 0:
                        continue

                    # process np raw to torch tensor
                    obj_id = meta['object_ids'][obj_index]
                    obj_model = obj_model_list[obj_id]
                    model, optimizer, lr_scheduler = obj_nn_models[obj_id]

                    # for debug
                    if obj_id != 0:
                        continue

                    # Run model forward
                    # transform as obj model (obj model in [-1, 1]^3)
                    world_coord = ((world_coord + obj_model.translate_to_0) * obj_model.scale_to_1).reshape(1, 3, -1)
                    model_coord = ((model_coord + obj_model.translate_to_0) * obj_model.scale_to_1).reshape(1, 3, -1)
                    world_coord = torch.tensor(world_coord, device=config.default_device).reshape(1, 3, -1).float()
                    model_coord = torch.tensor(model_coord, device=config.default_device).reshape(1, 3, -1).float()

                    out = model(world_coord)
                    a1 = out[0, :3]  # (3, 1)
                    a2 = out[0, 3:6]  # (3, 1)
                    t = out[0, 6:]  # (3, 1)
                    R = from_6Dpose_to_R(a1, a2)

                    # Compute loss
                    pred_model_in_world_coord = (torch.matmul(R, model_coord.reshape(3, -1)) - t).reshape(-1, 3)
                    # Fourier loss for rotation
                    # TODO
                    # loss for translation
                    # TODO
                    # half-CD loss for shape
                    loss_cd = config.loss_fn(pred_model_in_world_coord, world_coord)
                    loss = loss_cd

                    loss_batch += loss.item()
                    loss_n += 1

                    # Backprop
                    optimizer.zero_grad()
                    loss_cd.backward()
                    optimizer.step()

        t_range.set_description(f'Epoch {epoch:05d}')
        t_range.set_postfix(loss=loss_batch[0] / loss_n)

        # Adjust the learning rate
        for _, _, lr_scheduler in obj_nn_models:
            lr_scheduler.step()

        # Checkpoint
        if epoch % config.checkpoint_interval == 0 and epoch > 0:
            logger.info(f'Saving checkpoint {epoch}')
            for obj_id, (model, optimizer, lr_scheduler) in enumerate(obj_nn_models):
                save_state_dict(model, optimizer, epoch, obj_id)

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


def load_data(obj_model_list: ObjList, rgb: np.ndarray, depth: np.ndarray, label: np.ndarray, meta: dict):
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

    world_space_list = []
    model_space_list = []
    pose_world_list = []
    box_sizes_list = []

    for obj_id in meta['object_ids']:
        obj_model = obj_model_list[obj_id]
        scales = meta['scales'][obj_id].reshape(1, 3)
        model_coord = obj_model.points * scales

        if 'poses_world' not in meta:
            pose_world = None
        else:
            pose_world = meta['poses_world'][obj_id]
        box_sizes = meta['extents'][obj_id] * scales
        # logger.debug(f'camera_space.shape: {camera_space.shape}')
        # logger.debug(f'extrinsic.shape: {extrinsic.shape}')
        # exit(0)

        # find the position where label == index
        mask = np.abs(label - obj_id) < 1e-1
        # generate points from depth and mask
        world_coord = world_space[mask]

        world_space_list.append(world_coord)
        model_space_list.append(model_coord)
        pose_world_list.append(pose_world)
        box_sizes_list.append(box_sizes)

    return world_space_list, model_space_list, pose_world_list, box_sizes_list


def test(algo_type: str = 'icp', nn_info: Tuple = None):
    test_dataset = get_datasets('test')
    obj_csv_path = os.path.join(config.testing_data_dir, 'objects_v1.csv')
    obj_model_list = ObjList(obj_csv_path)

    output = {}
    n_all = 0
    n_correct = 0

    if os.path.exists(config.output_path) and os.path.exists(config.extra_info_path):
        output = json.load(open(config.output_path, 'r'))
        extra_info = pickle.load(open(config.extra_info_path, 'rb'))
        n_all = extra_info['n_all']
        n_correct = extra_info['n_correct']

    if algo_type == 'icp':
        with tqdm(range(len(test_dataset)), ncols=160) as pbar:
            for i in pbar:
                rgb, depth, label, meta, prefix = test_dataset[i]
                pbar.set_description(f'ICP {prefix:9s}')
                if prefix in output:
                    continue

                output[prefix] = {}
                pose_world_predict_list = [None for _ in range(config.n_obj)]

                for obj_index, (world_coord, model_coord, pose_world, box_sizes) in \
                        enumerate(zip(*load_data(obj_model_list, rgb, depth, label, meta))):
                    # skip if label map is wrong
                    if world_coord.shape[0] == 0:
                        continue

                    obj_id = meta['object_ids'][obj_index]
                    obj_model = obj_model_list[obj_id]

                    # try to match part of the model to the whole
                    R, t, loss = icp(world_coord, model_coord, obj_model, config.icp_max_iter, config.icp_tolerance)
                    R_inv = np.linalg.inv(R)
                    R, t = R_inv, -R_inv @ t

                    pose_world_pred = np.eye(4)
                    pose_world_pred[:3, :3] = R
                    pose_world_pred[:3, 3] = t
                    pose_world_predict_list[obj_id] = pose_world_pred.tolist()

                    # evaluate whether the prediction is correct
                    if pose_world is not None:
                        r_diff, t_diff = eval(pose_world_pred, pose_world, obj_model.geometric_symmetry)
                        n_correct += judge(r_diff, t_diff)

                    # # logger.info(f'------------------{prefix}------------------')
                    # # logger.info(f'obj_id = {obj_id}, obj_name = {obj_model.name}')
                    # # logger.info(f'loss = {loss:.06f}')
                    # # logger.info(f'pose_world_pred =\n{pose_world_pred}')
                    # # logger.info(f'pose_world =\n{pose_world} degree')
                    # # logger.info(f'geometric_symmetry = {obj_model.geometric_symmetry}')
                    # # logger.info(f"r_diff = {r_diff:.03f} degree, t_diff = {t_diff:.03f} cm")
                    # # exit(0)

                    n_all += 1
                    correct_rate = n_correct / n_all
                    pbar.set_postfix_str(f'correct ({n_correct}/{n_all}, {correct_rate:.06f})')

                    # # visualize
                    # if config.visualize:
                    #     R_ref = pose_world[:3, :3]
                    #     t_ref = pose_world[:3, 3]
                    #     visualize_point_cloud(world_coord, model_coord @ R.T + t, model_coord @ R_ref.T + t_ref)

                output[prefix]['poses_world'] = pose_world_predict_list

                if i % config.test_output_interval == 0 and i > 0:
                    json.dump(output, open(config.output_path, 'w'))
                    pickle.dump({'n_all': n_all, 'n_correct': n_correct}, open(config.extra_info_path, 'wb'))

        json.dump(output, open(config.output_path, 'w'))

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
            for iteration, batch in enumerate(tqdm.tqdm(test_dataloader, desc='NN', ncols=160)):
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
        if config.process == 'train':
            train()
        elif config.process == 'test':
            test('nn')
    elif config.algo_type == 'icp':
        test('icp')


if __name__ == "__main__":
    main()
