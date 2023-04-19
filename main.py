from configs.config_utilities import load_config
from torch.utils.data import DataLoader

from models.pyramid import build_pyramid_occupancy_network
from dataset import NuScenesDataset

import torch
import platform
import re

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
    
    images, labels, masks, calibs = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)
    calibs = calibs.to(device)

    predictions = network(images, calibs)
    print(predictions.shape)

if __name__ == "__main__":
    main()