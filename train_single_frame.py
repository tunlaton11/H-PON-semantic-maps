from configs.config_utilities import load_config
from dataset import NuScenesDataset
from model import UNET

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

import platform
import re
from tqdm import tqdm

import time
from logger import TensorboardLogger


def main():
    config = load_config()

    # first frame of first scene
    sample_tokens = ["e3d495d4ac534d54b321f50006683844"]

    train_dataset = NuScenesDataset(
        nuscenes_dir=config.nuscenes_dir,
        nuscenes_version=config.nuscenes_version,
        image_size=(200, 196),
        label_dir=config.label_dir,
        sample_tokens=sample_tokens,
        # flatten_labels=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=2,
        pin_memory=True,
        shuffle=True,
    )

    validate_dataset = NuScenesDataset(
        nuscenes_dir=config.nuscenes_dir,
        nuscenes_version=config.nuscenes_version,
        image_size=(200, 196),
        label_dir=config.label_dir,
        sample_tokens=sample_tokens,
        # flatten_labels=True,
    )
    validate_loader = DataLoader(
        validate_dataset,
        batch_size=1,
        num_workers=2,
        pin_memory=True,
        shuffle=True,
    )

    this_device = platform.platform()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = "cuda"
    elif re.search("arm64", this_device):
        # use Apple GPU
        device = "mps"
    else:
        device = "cpu"

    print(f"----- Training on {device} -----")

    network = UNET(in_channels=3, out_channels=14)
    network.to(device)

    if train_dataset.flatten_labels:
        loss_fn = nn.CrossEntropyLoss().to(device)
    else:
        loss_fn = nn.BCEWithLogitsLoss().to(device)

    optimizer = optim.Adam(network.parameters(), lr=config.lr)

    current_time = time.time()
    logger = TensorboardLogger(
        device,
        log_dir=f"{config.log_dir}/{current_time}",
        validate_loader=validate_loader,
        loss_fn=loss_fn,
        n_classes=14,
    )

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
            </tr>
            <tr>
                <td>{config.nuscenes_version}</td>
                <td>{train_dataset.transform is not None}</td>
                <td>{config.batch_size}</td>
                <td>{config.num_workers}</td>
                <td>{config.lr}</td>
                <td>{config.epochs}</td>
                <td>{device}</td>
                <td>{loss_fn.__class__.__name__}</td>
                <td>{optimizer.__class__.__name__}</td>
            </tr>
        </table>
    """

    logger.writer.add_text("Experiment Configurations", config_log_table, global_step=0)

    for epoch in tqdm(range(100)):
        for batch_idx, batch in enumerate(train_loader):
            image, labels, mask = batch
            image = image.to(device)

            if loss_fn.__class__.__name__ == "CrossEntropyLoss":
                labels = labels.long().to(device)

            else:
                labels = labels.type(torch.FloatTensor).to(device)

            mask = mask.to(device)

            prediction = network(image).to(device)

            # compute loss
            loss = loss_fn(prediction, labels).to(device)

            # compute gradient
            optimizer.zero_grad()
            loss.backward()

            # update weights
            optimizer.step()

            logger.log_step(loss=loss.item())

        logger.log_epoch(network, epoch)


if __name__ == "__main__":
    main()
