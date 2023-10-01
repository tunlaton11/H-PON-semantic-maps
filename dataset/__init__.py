from torch.utils.data import DataLoader
import albumentations as A
from .dataset import NuScenesDataset
from . import nuscenes_splits
# import numpy as np


def build_dataloaders(config) -> tuple[DataLoader, DataLoader]:
    """Build train and val dataloaders for NuScenes dataset"""

    print(f"Loading train dataset of NuScenes version {config.nuscenes_version}...")
    train_dataset = NuScenesDataset(
        nuscenes_dir=config.nuscenes_dir,
        nuscenes_version=config.nuscenes_version,
        label_dir=config.label_dir,
        scene_names=nuscenes_splits.TRAIN_SCENES,
        # sample_tokens=np.loadtxt("configs/mini_train_sample_tokens.csv", dtype=str),
        image_size=config.img_size,
        hflip=config.hflip,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=True,
    )
    print(f"Loading val dataset of NuScenes version {config.nuscenes_version}...")
    val_dataset = NuScenesDataset(
        nuscenes_dir=config.nuscenes_dir,
        nuscenes_version=config.nuscenes_version,
        label_dir=config.label_dir,
        scene_names=nuscenes_splits.VAL_SCENES,
        # sample_tokens=np.loadtxt("configs/mini_val_sample_tokens.csv", dtype=str),
        image_size=config.img_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=True,
    )

    return train_loader, val_loader
