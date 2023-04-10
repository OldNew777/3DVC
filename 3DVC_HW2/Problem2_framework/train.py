import torch
from torch.utils.data import DataLoader
from dataset import CubeDataset
from model import Img2PcdModel
from loss import CDLoss, HDLoss


def main():

    # TODO: Design the main function, including data preparation, training and evaluation processes.

    # Environment:
    # device: torch.device

    # Directories:
    # cube_data_path: str, cube dataset root directory
    # output_dir: str, result directory

    # Training hyper-parameters:
    # batch_size: int
    # epoch: int
    # learning_rate: float

    # Data lists:
    # training_cube_list: list
    # test_cube_list: list
    # view_idx_list: list

    # Preperation of datasets and dataloaders:
    # Example:
    #     training_dataset = CubeDataset(cube_data_path, training_cube_list, view_idx_list, device=device)
    #     test_dataset = CubeDataset(cube_data_path, test_cube_list, view_idx_list, device=device)
    #     training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    #     test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Network:
    # Example:
    #     model = Img2PcdModel(device=device)

    # Loss:
    # Example:
    #     loss_fn = CDLoss()

    # Optimizer:
    # Example:
    #     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training process:
    # Example:
    #     for epoch_idx in range(epoch):
    #         model.train()
    #         for batch_idx, (data_img, data_pcd) in enumerate(training_dataloader):
    #             # forward
    #             pred = model(data_img)

    #             # compute loss
    #             loss = loss_fn(pred, data_pcd)

    #             # backward
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
        
    # Final evaluation process:
    # Example:
    #     model.eval()
    #     for batch_idx, (data_img, data_pcd, data_r) in enumerate(test_dataloader):
    #         # forward
    #         pred = model(data_img, data_r)
    #         # compute loss
    #         loss = loss_fn(pred, data_pcd)
    
    pass


if __name__ == "__main__":
    main()
