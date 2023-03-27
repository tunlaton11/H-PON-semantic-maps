from configs.config_utilities import load_config
from dataset import NuScenesDataset
from model import UNET

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader


import platform
import re
from tqdm import tqdm

import time
from logger import TensorboardLogger


def main():
    config = load_config()
    train_dataset = NuScenesDataset(
        nuscenes_dir=config.nuscenes_dir,
        nuscenes_version=config.nuscenes_version,
        label_dir=config.label_dir,
        scene_names=config.train_scenes,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        num_workers=2,
        pin_memory=True,
        shuffle=True,
    )

    network = UNET(in_channels=3, out_channels=14)

    this_device = platform.platform()
    if torch.cuda.is_available():
        device = "cuda"
    elif re.search("arm64", this_device):
        # use Apple GPU
        device = "mps"
    else:
        device = "cpu"

    current_time = time.time()
    logger = TensorboardLogger(
        device,
        log_dir=f"{config.log_dir}/{current_time}",
    )

    print(f"----- Training on {device} -----")

    loss_fn = nn.CrossEntropyLoss().to(device)
    # loss_fn = nn.BCELoss()
    optimizer = optim.Adam(network.parameters(), lr=0.001)

    network.to(device)

    for epoch in tqdm(range(20)):
        # print(f"Training epoch {epoch+1}...")
        for batch_idx, batch in enumerate(train_loader):

            image, labels, mask = batch
            image = image.to(device)
            labels = labels.type(torch.FloatTensor).to(device)
            mask = mask.to(device)

            prediction = network(image).to(device)

            # print('pred', prediction.shape, type(prediction))
            # print('true label', labels.shape, type(labels))

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
