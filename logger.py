from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import NuSceneDataset
from configs.config_utilities import load_config


class TensorboardLogger:
    def __init__(self, device, log_dir):
        self.device = device
        self.writer = SummaryWriter(log_dir)

        self.training_step = 0
        self.training_loss = 0
        self.num_steps_per_epoch = 0

        config = load_config()

        validate_dataset = NuSceneDataset(
            nuscenes_dir=config.nuscenes_dir,
            nuscenes_version=config.nuscenes_version,
            label_dir=config.label_dir,
            start_scene_index=config.val_start_scene,
            end_scene_index=config.val_end_scene,
        )
        self.validate_loader = DataLoader(
            validate_dataset,
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

    def log_epoch(self, network: nn.Module):

        # Training
        self.writer.add_scalar(
            "Training/avg_training_loss",
            self.training_loss / self.num_steps_per_epoch,
            self.training_step,
        )

        self.training_loss = 0
        self.num_steps_per_epoch = 0
        self.validate(network)

    def validate(self, network: nn.Module):
        network.eval()  # set network's behavior to evaluation mode

        total_loss = 0
        num_step = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.validate_loader):

                image, labels, mask = batch
                image = image.to(self.device)
                labels = labels.type(torch.FloatTensor).to(self.device)
                mask = mask.to(self.device)

                prediction = network(image).to(self.device)
                loss = self.loss_fn(prediction, labels).to(self.device)
                total_loss += loss.item()
                num_step += 1

        self.writer.add_scalar(
            "Validate/avg_loss",
            total_loss / num_step,
            self.training_step,
        )
        network.train()  # set network's behavior to training mode
