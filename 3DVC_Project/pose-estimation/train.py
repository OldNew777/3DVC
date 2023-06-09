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


def save_state_dict(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, obj_id: int, scene: int):
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scene': scene,
    }
    path = checkpoint_path(epoch, obj_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)


def load_state_dict(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, obj_id: int):
    path = checkpoint_path(epoch, obj_id)
    state_dict = torch.load(path)
    model.load_state_dict(state_dict['model'])
    # optimizer.load_state_dict(state_dict['optimizer'])
    assert epoch == state_dict['epoch']
    return state_dict['scene']


def create_model(mode: str = 'train'):
    if mode != 'train' and mode != 'test':
        raise ValueError(f'Unknown mode {mode}')

    obj_nn_models = []

    # Load checkpoints
    start_epoch = checkpoint_epoch()
    start_scene = 0
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
            start_scene = load_state_dict(model, optimizer, start_epoch, obj_id)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

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

        obj_nn_models.append([model, optimizer, lr_scheduler, start_scene])

    return start_epoch, obj_nn_models


def train():
    start_epoch, obj_nn_models = create_model('train')
    start_scene = None
    n_multi_gpu = len(config.device_ids) if config.multi_gpu else 1

    train_dataset = get_datasets('train')
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=config.batch_size * n_multi_gpu,
    #     shuffle=True,
    #     num_workers=0,
    #     collate_fn=lambda batch: batch,
    # )
    validate_dataset = get_datasets('validate')
    # validate_dataloader = torch.utils.data.DataLoader(
    #     validate_dataset,
    #     batch_size=config.batch_size * n_multi_gpu,
    #     shuffle=False,
    #     num_workers=0,
    #     collate_fn=lambda batch: batch,
    # )
    obj_csv_path = os.path.join(config.training_data_dir, 'objects_v1.csv')
    obj_model_list = ObjList(obj_csv_path)

    t_range = tqdm(range(config.num_epochs), ncols=160)
    for epoch in t_range:
        if (epoch + 1) < start_epoch:
            continue

        loss_epoch = torch.zeros(1, device=config.default_device)
        loss_n = 0
        n_scene = len(train_dataset)
        for i in range(n_scene):
            if start_scene is not None and \
                    (i < start_scene and epoch == start_epoch) or \
                    (0 == start_scene and epoch + 1 == start_epoch):
                continue

            rgb, depth, label, meta, prefix = train_dataset[i]

            scene_str = f'{i}/{n_scene}'
            t_range.set_description(f'Epoch {epoch:05d}, Scene {scene_str:10s}')

            for obj_index, (world_coord, model_coord, pose_world, box_sizes) in \
                    enumerate(zip(*load_data(obj_model_list, rgb, depth, label, meta))):
                # skip if label map is wrong
                if world_coord.shape[0] == 0:
                    continue

                # process np raw to torch tensor
                obj_id = meta['object_ids'][obj_index]
                obj_model = obj_model_list[obj_id]
                world_coord_bak = world_coord.copy().T  # (3, N)
                model_coord_bak = model_coord.copy().T  # (3, N)

                model, optimizer, lr_scheduler, start_scene = obj_nn_models[obj_id]
                if (i < start_scene and epoch == start_epoch) or (0 == start_scene and epoch + 1 == start_epoch):
                    continue

                # Run model forward
                # transform as obj model (obj model in [-1, 1]^3)
                world_coord = (
                        (world_coord.T + obj_model.translate_to_0.reshape(3, 1)) * obj_model.scale_to_1)  # (3, N)
                world_coord_max = world_coord.max(axis=1, keepdims=True)
                world_coord_min = world_coord.min(axis=1, keepdims=True)
                world_coord_mid = (world_coord_max + world_coord_min) / 2
                world_coord = world_coord - world_coord_mid
                model_coord = (
                        (model_coord.T + obj_model.translate_to_0.reshape(3, 1)) * obj_model.scale_to_1)  # (3, N)
                world_coord = torch.tensor(world_coord, device=config.default_device).float()  # (3, N)
                model_coord = torch.tensor(model_coord, device=config.default_device).float()  # (3, N)

                out = model(world_coord.unsqueeze(0), model_coord.unsqueeze(0))  # (3, N)*2 -> (1, 3, N)*2 -> (1, 9, 1)
                a1 = out[0, :3]  # (3, 1)
                a2 = out[0, 3:6]  # (3, 1)
                t = out[0, 6:]  # (3, 1)
                R = from_6Dpose_to_R(a1, a2)

                t0 = torch.tensor(obj_model.translate_to_0.reshape(3, 1), device=config.default_device).float()
                s0 = torch.tensor(obj_model.scale_to_1, device=config.default_device).float()
                world_coord_mid = torch.tensor(world_coord_mid, device=config.default_device).float()
                t = R @ t0 - t0 + (t + world_coord_mid) / s0

                # Compute loss
                world_coord = torch.tensor(world_coord_bak, device=config.default_device).float()  # (3, N)
                model_coord = torch.tensor(model_coord_bak, device=config.default_device).float()  # (3, N)
                pred_model_in_world_coord = R @ model_coord - t  # (3, N)

                # # loss for transformation (geometry symmetric un-considered)
                # ref_6d = torch.concat([torch.tensor(pose_world[0, :3], device=config.default_device),
                #                        torch.tensor(pose_world[1, :3], device=config.default_device)], dim=0).squeeze().float()
                # pred_6d = torch.concat([a1, a2], dim=0).squeeze()
                # loss_6d = torch.linalg.norm(pred_6d - ref_6d) + \
                #           torch.linalg.norm(t.squeeze() - torch.tensor(pose_world[2, :3], device=config.default_device))
                # half-CD loss for shape
                loss_cd = config.loss_fn(world_coord.T, pred_model_in_world_coord.T)

                loss = loss_cd

                loss_epoch += loss
                loss_n += 1

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t_range.set_postfix(loss=loss.item(), loss_epoch=loss_epoch.item() / loss_n)
                # logger.info(f'Epoch {epoch:05d}, Scene {scene_str:10s}, Loss {loss_batch.item() / loss_n:.4f}')

                if config.visualize and (i + 1) % config.visualize_scene_interval == 0:
                    visualize_point_cloud(model_coord_bak.T, pred_model_in_world_coord.cpu().detach().numpy().T)

                if (i + 1) % config.checkpoint_interval_scene == 0:
                    logger.info(f'Saving checkpoint {epoch}, scene {i + 1}')
                    for obj_id, (model, optimizer, lr_scheduler, _) in enumerate(obj_nn_models):
                        save_state_dict(model, optimizer, epoch, obj_id, i + 1)

        # Adjust the learning rate
        for _, _, lr_scheduler, _ in obj_nn_models:
            lr_scheduler.step()

        # Checkpoint
        if (epoch + 1) % config.checkpoint_interval == 0:
            logger.info(f'Saving checkpoint {epoch + 1}')
            for obj_id, (model, optimizer, lr_scheduler, _) in enumerate(obj_nn_models):
                save_state_dict(model, optimizer, epoch + 1, obj_id, 0)

        # Validate
        if (epoch + 1) % config.validate_interval == 0 and epoch > 0:
            with torch.no_grad():
                n_all_validate = 0
                correct_n_validate = 0
                for i in range(len(validate_dataset)):
                    rgb, depth, label, meta, prefix = validate_dataset[i]

                    for obj_index, (world_coord, model_coord, pose_world, box_sizes) in \
                            enumerate(zip(*load_data(obj_model_list, rgb, depth, label, meta))):
                        # skip if label map is wrong
                        if world_coord.shape[0] == 0:
                            continue

                        # process np raw to torch tensor
                        obj_id = meta['object_ids'][obj_index]
                        obj_model = obj_model_list[obj_id]
                        world_coord_bak = world_coord.copy().T  # (3, N)
                        model_coord_bak = model_coord.copy().T  # (3, N)

                        model, optimizer, lr_scheduler, start_scene = obj_nn_models[obj_id]
                        if (i < start_scene and epoch == start_epoch) or (
                                0 == start_scene and epoch + 1 == start_epoch):
                            continue

                        # Run model forward
                        # transform as obj model (obj model in [-1, 1]^3)
                        world_coord = (
                                (world_coord.T + obj_model.translate_to_0.reshape(3,
                                                                                  1)) * obj_model.scale_to_1)  # (3, N)
                        world_coord_max = world_coord.max(axis=1, keepdims=True)
                        world_coord_min = world_coord.min(axis=1, keepdims=True)
                        world_coord_mid = (world_coord_max + world_coord_min) / 2
                        world_coord = world_coord - world_coord_mid
                        model_coord = (
                                (model_coord.T + obj_model.translate_to_0.reshape(3,
                                                                                  1)) * obj_model.scale_to_1)  # (3, N)
                        world_coord = torch.tensor(world_coord, device=config.default_device).float()  # (3, N)
                        model_coord = torch.tensor(model_coord, device=config.default_device).float()  # (3, N)

                        out = model(world_coord.unsqueeze(0),
                                    model_coord.unsqueeze(0))  # (3, N)*2 -> (1, 3, N)*2 -> (1, 9, 1)
                        a1 = out[0, :3]  # (3, 1)
                        a2 = out[0, 3:6]  # (3, 1)
                        t = out[0, 6:]  # (3, 1)
                        R = from_6Dpose_to_R(a1, a2)

                        t0 = torch.tensor(obj_model.translate_to_0.reshape(3, 1), device=config.default_device).float()
                        s0 = torch.tensor(obj_model.scale_to_1, device=config.default_device).float()
                        world_coord_mid = torch.tensor(world_coord_mid, device=config.default_device).float()
                        t = R @ t0 - t0 + (t + world_coord_mid) / s0

                        # Compute loss
                        n_all_validate += 1
                        pred_pose_world = np.eye(4)
                        pred_pose_world[:3, :3] = R.cpu().detach().numpy()
                        pred_pose_world[:3, 3] = t.cpu().detach().numpy().reshape(3)
                        r_diff, t_diff = eval(pred_pose_world, pose_world, obj_model.geometric_symmetry)
                        correct_n_validate += judge(r_diff, t_diff)

            logger.info(f'Epoch {epoch + 1:05d}, Validate correct: {correct_n_validate}/{n_all_validate}')


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


def test(algo_type: str = 'icp'):
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

                    # visualize
                    if config.visualize and pose_world is not None and not judge(r_diff, t_diff):
                        R_ref = pose_world[:3, :3]
                        t_ref = pose_world[:3, 3]
                        visualize_point_cloud(world_coord, model_coord @ R.T + t)

                output[prefix]['poses_world'] = pose_world_predict_list

                if i % config.test_output_interval == 0 and i > 0:
                    json.dump(output, open(config.output_path, 'w'))
                    pickle.dump({'n_all': n_all, 'n_correct': n_correct}, open(config.extra_info_path, 'wb'))

    elif algo_type == 'nn':
        _, obj_nn_models = create_model('train')

        # test_dataloader = torch.utils.data.DataLoader(
        #     test_dataset,
        #     batch_size=config.batch_size,
        #     shuffle=True,
        #     num_workers=0,
        #     collate_fn=lambda batch: batch,
        # )

        with torch.no_grad():
            n_all_validate = 0
            correct_n_test = 0
            pbar = tqdm(range(len(test_dataset)), ncols=160, desc='NN')
            for i in pbar:
                rgb, depth, label, meta, prefix = test_dataset[i]
                if prefix in output:
                    continue

                output[prefix] = {}
                pose_world_predict_list = [None for _ in range(config.n_obj)]

                for obj_index, (world_coord, model_coord, pose_world, box_sizes) in \
                        enumerate(zip(*load_data(obj_model_list, rgb, depth, label, meta))):
                    # skip if label map is wrong
                    if world_coord.shape[0] == 0:
                        continue

                    # process np raw to torch tensor
                    obj_id = meta['object_ids'][obj_index]
                    obj_model = obj_model_list[obj_id]
                    world_coord_bak = world_coord.copy().T  # (3, N)
                    model_coord_bak = model_coord.copy().T  # (3, N)

                    model, optimizer, lr_scheduler, start_scene = obj_nn_models[obj_id]

                    # Run model forward
                    # transform as obj model (obj model in [-1, 1]^3)
                    world_coord = (
                            (world_coord.T + obj_model.translate_to_0.reshape(3,
                                                                              1)) * obj_model.scale_to_1)  # (3, N)
                    world_coord_max = world_coord.max(axis=1, keepdims=True)
                    world_coord_min = world_coord.min(axis=1, keepdims=True)
                    world_coord_mid = (world_coord_max + world_coord_min) / 2
                    world_coord = world_coord - world_coord_mid
                    model_coord = (
                            (model_coord.T + obj_model.translate_to_0.reshape(3,
                                                                              1)) * obj_model.scale_to_1)  # (3, N)
                    world_coord = torch.tensor(world_coord, device=config.default_device).float()  # (3, N)
                    model_coord = torch.tensor(model_coord, device=config.default_device).float()  # (3, N)

                    out = model(world_coord.unsqueeze(0),
                                model_coord.unsqueeze(0))  # (3, N)*2 -> (1, 3, N)*2 -> (1, 9, 1)
                    a1 = out[0, :3]  # (3, 1)
                    a2 = out[0, 3:6]  # (3, 1)
                    t = out[0, 6:]  # (3, 1)
                    R = from_6Dpose_to_R(a1, a2)

                    t0 = torch.tensor(obj_model.translate_to_0.reshape(3, 1), device=config.default_device).float()
                    s0 = torch.tensor(obj_model.scale_to_1, device=config.default_device).float()
                    world_coord_mid = torch.tensor(world_coord_mid, device=config.default_device).float()
                    t = R @ t0 - t0 + (t + world_coord_mid) / s0
                    pose_world_pred = np.eye(4)
                    pose_world_pred[:3, :3] = R.cpu().detach().numpy()
                    pose_world_pred[:3, 3] = t.cpu().detach().numpy().reshape(3)
                    pose_world_predict_list[obj_id] = pose_world_pred.tolist()

                    # Compute loss
                    n_all_validate += 1
                    pred_pose_world = np.eye(4)
                    pred_pose_world[:3, :3] = R.cpu().detach().numpy()
                    pred_pose_world[:3, 3] = t.cpu().detach().numpy().reshape(3)
                    # r_diff, t_diff = eval(pred_pose_world, pose_world, obj_model.geometric_symmetry)
                    # correct_n_test += judge(r_diff, t_diff)

                output[prefix]['poses_world'] = pose_world_predict_list

                if i % config.test_output_interval == 0 and i > 0:
                    json.dump(output, open(config.output_path, 'w'))
                    pickle.dump({'n_all': n_all, 'n_correct': n_correct}, open(config.extra_info_path, 'wb'))

    else:
        raise ValueError(f'Unknown algo_type {algo_type}')

    json.dump(output, open(config.output_path, 'w'))
    pickle.dump({'n_all': n_all, 'n_correct': n_correct}, open(config.extra_info_path, 'wb'))


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
