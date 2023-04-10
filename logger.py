from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nuscenes_utilities import NUSCENES_CLASS_NAMES
from matplotlib.cm import get_cmap

# from typing import Literal, Callable
import torchmetrics.classification


class TensorboardLogger:
    def __init__(
        self,
        device: str,
        log_dir: str,
        validate_loader: DataLoader,
        loss_fn,  # Callable,
        n_classes: int,
        task="multilabel",  # Literal["multiclass", "multilabel"] = "multilabel",
        iou_average="macro",  # Literal["micro", "macro", "weighted", "none"] = "macro",
    ):
        self.device = device
        self.writer = SummaryWriter(log_dir)

        self.training_step = 0
        self.training_loss = 0
        self.num_steps_per_epoch = 0

        self.validate_loader = validate_loader
        self.loss_fn = loss_fn

        if task == "multiclass":
            num_classes = n_classes
            num_labels = None
        elif task == "multilabel":
            num_classes = None
            num_labels = n_classes

        self.iou_metric = torchmetrics.classification.JaccardIndex(
            task=task,
            num_classes=num_classes,
            num_labels=num_labels,
            average=iou_average,
        )

    def log_step(self, loss: float):
        self.training_loss += loss
        self.training_step += 1
        self.num_steps_per_epoch += 1

    def log_epoch(self, network: nn.Module, epoch: int):
        # Training
        self.writer.add_scalar(
            "Train/avg_loss",
            self.training_loss / self.num_steps_per_epoch,
            self.training_step,
        )

        self.training_loss = 0
        self.num_steps_per_epoch = 0
        self.validate(network, epoch)

    def validate(self, network: nn.Module, epoch: int):
        network.eval()  # set network's behavior to evaluation mode

        total_loss = 0
        total_iou = 0
        num_step = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.validate_loader):
                image, labels, mask = batch
                image = image.to(self.device)
                labels = labels.type(torch.FloatTensor).to(self.device)
                mask = mask.to(self.device)

                prediction = network(image).to(self.device)
                # prediction = prediction.sigmoid()
                loss = self.loss_fn(prediction, labels).to(self.device)
                total_loss += loss.item()
                # iou = self.iou_metric(prediction.IntTensor(), labels)
                # total_iou += iou
                num_step += 1

        visualise(
            self.writer, image, prediction, labels, mask, epoch, "nuscenes", split="val"
        )

        self.writer.add_scalar(
            "Validate/avg_loss",
            total_loss / num_step,
            self.training_step,
        )
        self.writer.add_scalar(
            "Validate/avg_iou",
            total_iou / num_step,
            self.training_step,
        )

        network.train()  # set network's behavior to training mode


def colorise(tensor, cmap, vmin=None, vmax=None):
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
    scores,
    labels,
    mask,
    step,
    dataset,
    split,
):
    class_names = NUSCENES_CLASS_NAMES

    summary.add_image(split + "/image", image[0], step, dataformats="CHW")
    summary.add_image(
        split + "/pred", colorise(scores[0], "coolwarm", 0, 1), step, dataformats="NHWC"
    )
    summary.add_image(
        split + "/gt", colorise(labels[0], "coolwarm", 0, 1), step, dataformats="NHWC"
    )


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
