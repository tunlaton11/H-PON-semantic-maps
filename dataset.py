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
        image_size:Tuple[int, int]=(200, 196),
        transform:A.Compose=None,
        image_transform:A.Compose=None,
        scene_names:Iterable[str]=None,
        flatten_labels=False,
    ):
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
        self.get_tokens(scene_names)

    def get_tokens(
        self,
        scene_names:Iterable[str]=None,
    ):
        self.tokens = list()

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

        return self.tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index: int):
        token = self.tokens[index]
        image = self.load_image(token)
        labels, mask = self.load_labels(token)

        if self.flatten_labels:
            labels = labels.detach().numpy()
            labels = nusc_utils.flatten_labels(labels)

        if self.transform is not None:
            augmentations = self.transform(
                image=image,
                labels=labels,
                mask=mask,
            )
            image = augmentations["image"]
            labels = augmentations["labels"]
            mask = augmentations["mask"]
        
        if self.image_transform is not None:
            augmentations = self.image_transform(image=image)
            image = augmentations["image"]

        # Convert to torch tensor
        return (
            to_tensor(image),
            torch.from_numpy(labels),
            torch.from_numpy(mask),
        )

    def load_image(self, token: str):

        # Load image
        image = cv2.imread(self.nuscenes.get_sample_data_path(token))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to input resolution
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


if __name__ == "__main__":

    dataset = NuScenesDataset(
        nuscenes_dir="nuscenes", nuscenes_version="v1.0-mini", label_dir="labels"
    )

    image, labels, mask = dataset[0]
    print(labels[0])
    print(labels.shape)
