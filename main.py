import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
import numpy as np
from tqdm import tqdm
import os
import time

from configs.config_utilities import load_config
from models.pyramid import build_pon, build_hpon
from dataset import NuScenesDataset
import nuscenes_splits

from criterion import OccupancyCriterion
from logger import TensorboardLogger
import utilities.torch as torch_utils


def main():
    ## SET EXPERIMENT CONFIG ##
    config = load_config("configs/configs.yml")
    experiment_title = f"{config.network}_{config.nuscenes_version}_{time.time()}"
    is_load_checkpoint = False
    if is_load_checkpoint:
        # set experiment title and checkpoint path manuanlly here.
        experiment_title = "[network]_[nuscenes_version]_[unix_timestamp]"
        load_checkpoint_epoch = 99  # epoch to load checkpoint
        load_checkpoint_path = (
            f"{config.checkpoint_dir}/{experiment_title}"
            + f"/{experiment_title}_{str(epoch).zfill(load_checkpoint_epoch)}.pt"
        )

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
        scene_names=nuscenes_splits.TRAIN_SCENES,
        # sample_tokens=np.loadtxt("configs/mini_train_sample_tokens.csv", dtype=str),
        image_size=(200, 112),
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
        scene_names=nuscenes_splits.VAL_SCENES,
        # sample_tokens=np.loadtxt("configs/mini_val_sample_tokens.csv", dtype=str),
        image_size=(200, 112),
    )
    validate_loader = DataLoader(
        validate_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=True,
    )

    device = torch_utils.detect_device()
    print(f"Training on {device}")

    if config.network == "H-PON":
        network = build_hpon(config, htfm_method="stack").to(device)
    elif config.network == "PON":
        network = build_pon(config).to(device)
    else:
        raise "Only H-PON and PON options available for network"

    # criterion = nn.BCEWithLogitsLoss().to(device)
    criterion = OccupancyCriterion(
        config.prior,
        config.xent_weight,
        config.uncert_weight,
        config.weight_mode,
    ).to(device)
    num_classes = 14

    optimizer = optim.Adam(network.parameters(), lr=config.lr)

    log_dir = f"{config.log_dir}/{experiment_title}"
    num_epochs = config.epochs
    if is_load_checkpoint:
        print(f"Loading checkpoint from {load_checkpoint_path}")
        checkpoint = torch.load(load_checkpoint_path)
        network.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        initial_step = checkpoint["step"]
        initial_epoch = checkpoint["epoch"] + 1
    else:
        initial_step = 0
        initial_epoch = 0

    logger = TensorboardLogger(
        device=device,
        log_dir=log_dir,
        validate_loader=validate_loader,
        criterion=criterion,
        num_classes=num_classes,
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

    for epoch in tqdm(range(initial_epoch, num_epochs)):
        for batch in train_loader:
            images, labels, masks, calibs = batch
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            calibs = calibs.to(device)

            logits = network(images, calibs)

            # compute loss
            # loss = criterion(predictions, labels.float()).to(device)
            loss = criterion(logits, labels, masks).to(device)

            # compute gradient
            optimizer.zero_grad()
            loss.backward()

            # update weights
            optimizer.step()

            logger.log_step(loss=loss.item())

        logger.log_epoch(network, epoch)

        # save checkpoint every n epochs
        if (
            epoch % config.num_epochs_to_save_checkpoint
            == config.num_epochs_to_save_checkpoint - 1
        ):
            print(f"Saving model at epoch {epoch}")
            checkpoint_dir = f"{config.checkpoint_dir}/{experiment_title}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = (
                checkpoint_dir + f"/{experiment_title}_{str(epoch).zfill(5)}.pt"
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

    # save last epoch
    if (
        epoch % config.num_epochs_to_save_checkpoint
        != config.num_epochs_to_save_checkpoint - 1
    ):
        print(f"Saving model at epoch {epoch}")
        checkpoint_dir = f"{config.checkpoint_dir}/{experiment_title}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = (
            checkpoint_dir + f"/{experiment_title}_{str(epoch).zfill(5)}.pt"
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
