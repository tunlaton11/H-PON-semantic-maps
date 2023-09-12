from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from nuscenes_utilities import NUSCENES_CLASS_NAMES, flatten_labels
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap

# from typing import Literal, Callable
import torchmetrics.classification
import numpy as np
import matplotlib.pyplot as plt
from experiments.ipm.ipm_utilities import ipm_transform

import torchvision.utils


class TensorboardLogger:
    def __init__(
        self,
        device: str,
        log_dir: str,
        validate_loader: DataLoader,
        criterion,  # Callable,
        num_classes: int,
        initial_step: int = 0,
    ):
        self.device = device
        self.writer = SummaryWriter(log_dir)

        self.training_step = initial_step
        self.training_loss = 0
        self.num_steps_per_epoch = 0

        self.validate_loader = validate_loader
        self.criterion = criterion

        self.iou_metric = torchmetrics.classification.MultilabelJaccardIndex(
            num_labels=num_classes,
            average="macro",
        ).to(device)
        self.iou_metric_by_class = torchmetrics.classification.MultilabelJaccardIndex(
            num_labels=num_classes,
            average="none",
        ).to(device)

        self.min_loss = float("inf")
        self.no_improve_consec_counter = 0
        self.save_model = False

    def log_step(self, loss: float):
        self.training_loss += loss
        self.training_step += 1
        self.num_steps_per_epoch += 1

    def log_epoch(self, network: nn.Module, epoch: int):
        # Training
        self.writer.add_scalar(
            "Train/avg_loss",
            self.training_loss / self.num_steps_per_epoch,
            epoch,
        )

        self.training_loss = 0
        self.num_steps_per_epoch = 0
        self.validate(network, epoch)

    def validate(self, network: nn.Module, epoch: int):
        network.eval()  # set network's behavior to evaluation mode

        total_loss = 0
        total_iou = 0
        total_iou_by_class = torch.zeros(14).to(self.device)
        num_step = 0

        with torch.no_grad():
            for batch in self.validate_loader:
                images, labels, masks, calibs = batch

                labels = labels.to(self.device)
                masks = masks.to(self.device)
                calibs = calibs.to(self.device)

                images = images.to(self.device)
                predictions = network(images, calibs).to(self.device)

                # predictions = network(images).to(self.device)

                loss = self.criterion(predictions, labels.float()).to(self.device)

                total_loss += loss.item()
                iou = self.iou_metric(predictions, labels)
                iou_by_class = self.iou_metric_by_class(predictions, labels)
                total_iou += iou
                total_iou_by_class += iou_by_class
                num_step += 1

        if total_loss < self.min_loss:
            self.min_loss = total_loss
            self.not_improve_consec_counter = 0
            self.save_model = True
        else:
            self.not_improve_consec_counter += 1
            self.save_model = False

        
        visualise(
            self.writer,
            images[-1],
            predictions[-1],
            labels[-1],
            masks[-1],
            epoch,
            "nuscenes",
            split="Validate",
        )

        for class_iou, class_name in zip(total_iou_by_class, NUSCENES_CLASS_NAMES):
            self.writer.add_scalar(
                f"Validate/iou/{class_name}",
                class_iou / num_step,
                epoch,
            )

        self.writer.add_scalar(
            "Validate/avg_loss",
            total_loss / num_step,
            epoch,
        )
        self.writer.add_scalar(
            "Validate/avg_iou",
            total_iou / num_step,
            epoch,
        )

        network.train()  # set network's behavior to training mode


def colorise(tensor, cmap, vmin=None, vmax=None, flatten=False):
    if flatten:
        cmap = get_cmap(cmap, 100)
        cmap_colors = cmap(np.linspace(0, 1, 15))[:14]
        cmap = ListedColormap(cmap_colors)

        class_prediction_color = cmap(tensor.cpu())
        class_prediction_color = class_prediction_color[..., :3]

        class_prediction_color = (
            torch.from_numpy(class_prediction_color).permute(2, 0, 1).unsqueeze(0)
        )

        return class_prediction_color

    else:
        if isinstance(cmap, str):
            cmap = get_cmap(cmap)

        tensor = tensor.detach().cpu().float()

        vmin = float(tensor.min()) if vmin is None else vmin
        vmax = float(tensor.max()) if vmax is None else vmax

        tensor = (tensor - vmin) / (vmax - vmin)
        return cmap(tensor.numpy())[..., :3]


def visualise(
    summary: SummaryWriter,
    image,
    pred,
    labels,
    mask,
    step,
    split,
):
    # class_names = NUSCENES_CLASS_NAMES

    colorised_pred = torch.from_numpy(
        colorise(pred.sigmoid(), "coolwarm", 0, 1)
    ).permute(0, 3, 1, 2)
    colorised_gt = torch.from_numpy(colorise(labels, "coolwarm", 0, 1)).permute(
        0, 3, 1, 2
    )

    pred = (pred.sigmoid() >= 0.5).long()

    colorised_flatten_gt = colorise(
        flatten_labels(labels.cpu()),
        "nipy_spectral",
        flatten=True,
    )
    colorised_flatten_pred = colorise(
        flatten_labels(pred.cpu()), "nipy_spectral", flatten=True
    )

    mask = (mask.cpu() == -1).long()

    gt_with_mask = colorised_flatten_gt * mask
    pred_with_mask = colorised_flatten_pred * mask

    colorised_pred = torch.cat(
        (colorised_pred, colorised_flatten_pred, pred_with_mask), dim=0
    )
    colorised_gt = torch.cat((colorised_gt, colorised_flatten_gt, gt_with_mask), dim=0)

    gt_grid = torchvision.utils.make_grid(colorised_gt, 6, 3)
    pred_grid = torchvision.utils.make_grid(colorised_pred, 6, 3)

    summary.add_image(split + "/image", image, step, dataformats="CHW")
    summary.add_image(
        split + "/predicted",
        pred_grid,
        step,
    )
    summary.add_image(split + "/gt", gt_grid, step)


def visualize_muticlass(
    writer: SummaryWriter,
    image,
    pred,
    labels,
    step,
    split,
):
    gt_to_colorise = labels

    labels = F.one_hot(labels, num_classes=15).permute((2, 0, 1))

    colorised_pred = torch.from_numpy(
        colorise(pred.softmax(dim=0), "coolwarm", 0, 1)
    ).permute(0, 3, 1, 2)
    colorised_gt = torch.from_numpy(colorise(labels, "coolwarm", 0, 1)).permute(
        0, 3, 1, 2
    )

    pred = (pred.softmax(dim=0) >= 0.5).long()

    colorised_flatten_gt = colorise(gt_to_colorise.cpu(), "viridis", flatten=True)
    colorised_flatten_pred = colorise(
        flatten_labels(pred.cpu()), "viridis", flatten=True
    )

    # concat with colorised flatten
    colorised_pred = torch.cat((colorised_pred, colorised_flatten_pred), dim=0)
    colorised_gt = torch.cat((colorised_gt, colorised_flatten_gt), dim=0)

    gt_grid = torchvision.utils.make_grid(colorised_gt[1:])
    img_grid = torchvision.utils.make_grid(colorised_pred[1:])

    writer.add_image(f"{split}/image", image, step)
    writer.add_image(f"{split}/gt", gt_grid, step)
    writer.add_image(f"{split}/predicted", img_grid, step)


def evaluate_preds(
    preds: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
    task,  # Literal["multiclass", "multilabel"],
    average,  # Literal["micro", "macro", "weighted", "none"] = "macro",
):
    """Evaluate the predictions for IoU, precision and recall.

    Parameters
    ----------
    preds : float tensor of shape
    (batch_size, n_classes, height, width)
    labels : int tensor of shape
    (batch_size, height, width)
    n_classes : int
        Number of classes
    task : 'multiclass' or 'multilabel'
    average : 'micro', 'macro', 'weighted', or 'none'
        Average calculation method.

    Returns
    -------
    iou : tensor float
    precision : tensor float
    recall : tensor float
    """

    if task == "multiclass":
        num_classes = n_classes
        num_labels = None
    elif task == "multilabel":
        num_classes = None
        num_labels = n_classes

    iou_metric = torchmetrics.classification.JaccardIndex(
        task=task,
        num_classes=num_classes,
        num_labels=num_labels,
        average=average,
    )
    iou = iou_metric(preds, labels)
    precision_metric = torchmetrics.classification.Precision(
        task=task,
        num_classes=num_classes,
        num_labels=num_labels,
        average=average,
    )
    precision = precision_metric(preds, labels)
    recall_metric = torchmetrics.classification.Recall(
        task=task,
        num_classes=num_classes,
        num_labels=num_labels,
        average=average,
    )
    recall = recall_metric(preds, labels)

    return iou, precision, recall
