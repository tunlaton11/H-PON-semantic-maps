from torch.utils.tensorboard import SummaryWriter
from dataset import NuSceneDataset
from torch.utils.data import DataLoader
import torch.nn as nn

from configs.config_utilities import load_config


class TensorboardLogger:
    def __init__(self, device, log_dir):
        self.device = device
        self.writer = SummaryWriter(log_dir)

        self.training_step = 0
        self.training_loss = 0
        self.num_steps_per_epoch = 0

        config = load_config()

        val_dataset = NuSceneDataset(
            nuscenes_dir=config.nuscenes_dir,
            nuscenes_version=config.nuscenes_version,
            label_dir=config.label_dir,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=2,
            num_workers=2,
            pin_memory=True,
            shuffle=True,
        )
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

    def log_step(self, loss):
        self.training_loss += loss
        self.training_step += 1
        self.num_steps_per_epoch += 1

    def log_epoch(self):

        # Training
        self.writer.add_scalar(
            "Training/avg_training_loss",
            self.training_loss / self.num_steps_per_epoch,
            self.training_step,
        )

        self.training_loss = 0
        self.num_steps_per_epoch = 0
