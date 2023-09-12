from configs.config_utilities import load_config
from torch.utils.data import DataLoader

from models.pyramid import (
    build_pyramid_occupancy_network,
    build_extended_pyramid_occupancy_network,
)
from dataset import NuScenesDataset

# from logger import TensorboardLogger
from logger_with_early_stop import TensorboardLogger
import utilities.torch as torch_utils

import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2

import os
import time
from tqdm import tqdm
import numpy as np


def main():
    config = load_config("configs/configs.yml")

    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
        ]
    )

    train_image_transform = A.Compose(
        [
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.2,
                p=0.25,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
        ]
    )

    train_dataset = NuScenesDataset(
        nuscenes_dir=config.nuscenes_dir,
        nuscenes_version=config.nuscenes_version,
        label_dir=config.label_dir,
        # sample_tokens=config.train_tokens,
        sample_tokens=np.loadtxt("configs/mini_train_sample_tokens.csv", dtype=str),
        image_size=(200, 112),
        flatten_labels=(config.method_type == "multiclass"),
        transform=train_transform,
        image_transform=train_image_transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=True,
    )
    validate_dataset = NuScenesDataset(
        nuscenes_dir=config.nuscenes_dir,
        nuscenes_version=config.nuscenes_version,
        label_dir=config.label_dir,
        # sample_tokens=config.val_tokens,
        sample_tokens=np.loadtxt("configs/mini_val_sample_tokens.csv", dtype=str),
        image_size=(200, 112),
        flatten_labels=(config.method_type == "multiclass"),
    )
    validate_loader = DataLoader(
        validate_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=True,
    )

    device = torch_utils.detect_device()

    # network = build_pyramid_occupancy_network(config).to(device)
    network = build_extended_pyramid_occupancy_network(config, htfm_method="stack").to(device)

    if train_dataset.flatten_labels:
        criterion = nn.CrossEntropyLoss().to(device)  # multiclass
        num_class = 15
        task = "multiclass"
    else:
        criterion = nn.BCEWithLogitsLoss().to(device)  # multilabel
        num_class = 14
        task = "multilabel"

    optimizer = optim.Adam(network.parameters(), lr=config.lr)



    is_load_checkpoint = False
    if is_load_checkpoint:
        experiment_title = "Full_EPON_H-collage_1693459089.077859"
        log_dir = f"runs/{experiment_title}"
        checkpoint_path = f"checkpoints/{experiment_title}/Full_EPON_H-collage_1693459089.077859_00099.pt"
        checkpoint = torch.load(checkpoint_path)
        network.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        initial_step = checkpoint["step"]
        initial_epoch = checkpoint["epoch"] + 1
        epochs = initial_epoch + 200
    else:
        current_time = time.time()
        # experiment_title = f"Full_EPON_H-collage_{current_time}"
        experiment_title = f"Full_EPON_H-stack_{current_time}"
        # experiment_title = f"Full_PON_{current_time}"
        log_dir = f"{config.log_dir}/{experiment_title}"
        initial_step = 0
        initial_epoch = 0
        epochs = config.epochs

    network.to(device)

    logger = TensorboardLogger(
        device,
        log_dir=log_dir,
        validate_loader=validate_loader,
        criterion=criterion,
        n_classes=num_class,
        task=task,
        initial_step=initial_step,
    )

    if not is_load_checkpoint:
        config_log_table = f"""
            <table>
                <tr>
                    <th>Nuscenes Version</th>
                    <th>Is augmentation</th>
                    <th>Batch Size</th>
                    <th>Num Workers</th>
                    <th>Learning Rate</th>
                    <th>Number of epochs</th>
                    <th>Device</th>
                    <th>Loss function</th>
                    <th>Optimizer</th>
                    <th>Network</th>
                </tr>
                <tr>
                    <td>{config.nuscenes_version}</td>
                    <td>{train_dataset.image_transform is not None}</td>
                    <td>{config.batch_size}</td>
                    <td>{config.num_workers}</td>
                    <td>{config.lr}</td>
                    <td>{config.epochs}</td>
                    <td>{device}</td>
                    <td>{criterion.__class__.__name__}</td>
                    <td>{optimizer.__class__.__name__}</td>
                    <td>{network.__class__.__name__}</td>
                </tr>
            </table>
        """
        logger.writer.add_text(
            "Experiment Configurations", config_log_table, global_step=0
        )

    for epoch in tqdm(range(initial_epoch, epochs)):
        for batch_idx, batch in enumerate(train_loader):
            images, labels, masks, calibs = batch
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            calibs = calibs.to(device)

            predictions = network(images, calibs)

            # compute loss
            if criterion.__class__.__name__ == "CrossEntropyLoss":
                loss = criterion(predictions, labels.long()).to(device)
            else:
                loss = criterion(predictions, labels.float()).to(device)

            # compute gradient
            optimizer.zero_grad()
            loss.backward()

            # update weights
            optimizer.step()

            logger.log_step(loss=loss.item())

        logger.log_epoch(network, epoch)

        # save best model
        # if logger.save_model:
        #     print(f"save model at epoch {epoch}")
        #     checkpoint_dir = os.path.expandvars(config.checkpoint_dir + "/" + experiment_title)
        #     os.makedirs(checkpoint_dir, exist_ok=True)

        #     checkpoint_path = (
        #         checkpoint_dir + f"/{experiment_title}_{str(epoch).zfill(5)}_best.pt"
        #     )

        #     torch.save(
        #         dict(
        #             epoch=epoch,
        #             step=logger.training_step,
        #             model_state_dict=network.state_dict(),
        #             optimizer_state_dict=optimizer.state_dict(),
        #         ),
        #         checkpoint_path,
        #     )

        # early stop
        # if logger.not_improve_consec_counter == 10:
        #     print(f"stop training at epoch {epoch}")
        #     break

    # save last epoch
    checkpoint_dir = os.path.expandvars(config.checkpoint_dir + "/" + experiment_title)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = (
        checkpoint_dir
        + f"/{experiment_title}_{str(epoch).zfill(5)}.pt"
    )
    torch.save(
        dict(
            epoch=epoch,
            step=logger.training_step,
            model_state_dict=network.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
        ),
        checkpoint_path,
    )


if __name__ == "__main__":
    main()
