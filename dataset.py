import os
from typing import Iterable, Tuple
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch
from torchvision.transforms.functional import to_tensor
import albumentations as A


from nuscenes import NuScenes
import nuscenes_utilities as nusc_utils


class NuScenesDataset(Dataset):
    def __init__(
        self,
        nuscenes_dir: str,
        nuscenes_version: str,
        label_dir: str,
        sample_tokens: Iterable[str] = None,
        scene_names: Iterable[str] = None,
        image_size: Tuple[int, int] = None,
        transform: A.Compose = None,
        image_transform: A.Compose = None,
        flatten_labels=False,
    ):
        """NuScenes Dataset.

        Parameters
        ----------
        nuscenes_dir : str
            Path of NuScenes data directory.
        nuscenes_version : str
            Version of NuScenes to load e.g. "v1.0-mini".
        label_dir : str
            Path of label directory.
        sample_tokens : list of str, optional
            List of sample (frame) tokens for the dataset, If None,
            the dataset includes all samples. Do not set both
            `sample_tokens` and `scene_names` at the same time.
        scene_names : list of str, optional
            List of scene names for the dataset. If None, the dataset
            includes all scenes.
        image_size : (width, height), optional
            Size of image.
        transform : A.Compose, optional
            Albumentations Compose Transform for image, label, and mask.
            e.g. horizontal flip. Do not include ToTensorV2 in transform
            as the dataset has its own torch tensor convert.
        image_transform : A.Compose, optional
            Albumentations Compose Transform for image.
            e.g. brightness, contrast, saturation. Do not include
            ToTensorV2 in transform as the dataset has its own torch
            tensor convert.
        flatten_labels : bool, optional
            If true, labels are flatten to one channel instead of
            n_classes channels. Default: False.
        """
        print("-" * 50)
        print(f"Loading NuScenes version {nuscenes_version} ...")
        self.nuscenes = NuScenes(
            nuscenes_version,
            nuscenes_dir,
            verbose=False,
        )
        print("-" * 50)
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.image_transform = image_transform
        self.flatten_labels = flatten_labels
        self.get_tokens(sample_tokens, scene_names)

    def get_tokens(
        self,
        sample_tokens: Iterable[str] = None,
        scene_names: Iterable[str] = None,
    ):
        self.tokens = list()

        if sample_tokens is None:
            # Iterate over scenes
            for scene in self.nuscenes.scene:
                # # Ignore scenes which don't belong to the current split
                if scene_names is not None and scene["name"] not in scene_names:
                    continue

                # Iterate over samples
                for sample in nusc_utils.iterate_samples(
                    self.nuscenes, scene["first_sample_token"]
                ):
                    self.tokens.append(sample["data"]["CAM_FRONT"])
        else:
            self.tokens = sample_tokens

        return self.tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index: int):
        token = self.tokens[index]
        image = self.load_image(token)
        labels, mask = self.load_labels(token)
        calib = self.load_calib(token)

        if self.flatten_labels:
            labels = nusc_utils.flatten_labels(labels)

        if self.transform is not None:
            augmentations = self.transform(
                image=image,
                labels=labels,
            )
            image = augmentations["image"]
            labels = augmentations["labels"]

        if self.image_transform is not None:
            augmentations = self.image_transform(image=image)
            image = augmentations["image"]

        # Convert to torch tensor
        return (
            to_tensor(image),
            torch.from_numpy(labels),
            torch.from_numpy(mask),
            calib
        )

    def load_image(self, token: str):
        # Load image
        image = cv2.imread(self.nuscenes.get_sample_data_path(token))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to input resolution
        if self.image_size is not None:
            image = cv2.resize(image, self.image_size)

        return image

    def load_labels(self, token: str):
        # Load label image
        label_path = os.path.join(self.label_dir, token + ".png")
        encoded_labels = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        # Decode to binary labels
        num_class = len(nusc_utils.NUSCENES_CLASS_NAMES)
        labels = nusc_utils.decode_binary_labels(encoded_labels, num_class + 1)
        labels, mask = labels[:-1], ~labels[-1]

        return labels, mask

    
    def load_calib(self, token):

        # Load camera intrinsics matrix
        sample_data = self.nuscenes.get('sample_data', token)
        sensor = self.nuscenes.get(
            'calibrated_sensor', sample_data['calibrated_sensor_token'])
        intrinsics = torch.tensor(sensor['camera_intrinsic'])

        # Scale calibration matrix to account for image downsampling
        intrinsics[0] *= self.image_size[0] / sample_data['width']
        intrinsics[1] *= self.image_size[1] / sample_data['height']
        return intrinsics


if __name__ == "__main__":
    dataset = NuScenesDataset(
        nuscenes_dir="nuscenes",
        nuscenes_version="v1.0-mini",
        label_dir="labels",
    )

    image, labels, mask = dataset[0]
    print(labels[0])
    print(labels.shape)
