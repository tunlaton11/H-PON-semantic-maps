from configs.config_utilities import load_config
from torch.utils.data import DataLoader

from models.pyramid import build_pyramid_occupancy_network
from dataset import NuScenesDataset
from logger import TensorboardLogger

import torch
import torch.nn as nn
import torch.optim as optim

import platform
import re
import time
from tqdm import tqdm


def main():
    config = load_config()

    sample_tokens = ["e3d495d4ac534d54b321f50006683844"]
    train_dataset = NuScenesDataset(
        nuscenes_dir=config.nuscenes_dir,
        nuscenes_version=config.nuscenes_version,
        label_dir=config.label_dir,
        sample_tokens=sample_tokens,
        # scene_names=config.train_scenes,
        image_size=(200, 196),
        flatten_labels=True,
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
        sample_tokens=sample_tokens,
        # scene_names=config.val_scenes,
        image_size=(200, 196),
        flatten_labels=True,
    )
    validate_loader = DataLoader(
        validate_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
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

    network = build_pyramid_occupancy_network().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(network.parameters(), lr=config.lr)

    current_time = time.time()
    logger = TensorboardLogger(
        device,
        log_dir=f"{config.log_dir}/{current_time}",
        validate_loader=validate_loader,
        criterion=criterion,
        n_classes=15,
        task="multiclass",
    )

    for epoch in tqdm(range(config.epochs)):
        for batch_idx, batch in enumerate(train_loader):
            images, labels, masks, calibs = batch
            images = images.to(device)
            labels = labels.to(device)
            calibs = calibs.to(device)

            predictions = network(images, calibs)
            # compute loss
            loss = criterion(predictions, labels.long()).to(device)

            # compute gradient
            optimizer.zero_grad()
            loss.backward()

            # update weights
            optimizer.step()

            logger.log_step(loss=loss.item())

        logger.log_epoch(network, epoch)


if __name__ == "__main__":
    main()
