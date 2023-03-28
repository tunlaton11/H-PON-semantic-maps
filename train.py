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

    train_transform = A.Compose(
        [
            A.Resize(height=196, width=200),
            A.Rotate(limit=35, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    train_dataset = NuScenesDataset(
        nuscenes_dir=config.nuscenes_dir,
        nuscenes_version=config.nuscenes_version,
        label_dir=config.label_dir,
        scene_names=config.train_scenes,
        # transform=train_transform,
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
        torch.cuda.empty_cache()
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

    # loss_fn = nn.CrossEntropyLoss().to(device)
    loss_fn = nn.BCELoss().to(device)
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
            prediction = prediction.sigmoid()

            # print("pred", prediction.shape, type(prediction))
            # break
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
