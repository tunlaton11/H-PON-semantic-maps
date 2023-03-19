import os
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from PIL import Image
import numpy as np
from nuscenes import NuScenes

import nuscenes_utilities as nusc_utils
from torch.utils.data import DataLoader


class NuSceneDataset(Dataset):
    def __init__(self, data_root, label_dir, image_size=(200, 196), transform=None):
        self.nuscenes = NuScenes("v1.0-mini", data_root)
        self.label_dir = label_dir
        self.image_size = image_size
        self.get_tokens()

    def get_tokens(self, scene_names=None):

        self.tokens = list()

        # Iterate over scenes
        for scene in self.nuscenes.scene[:1]:

            # Ignore scenes which don't belong to the current split
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

    def __getitem__(self, index):
        token = self.tokens[index]
        image = self.load_image(token)
        labels, mask = self.load_labels(token)
        return image, labels, mask

    def load_image(self, token):

        # Load image as a PIL image
        image = Image.open(self.nuscenes.get_sample_data_path(token))

        # Resize to input resolution
        image = image.resize(self.image_size)

        # Convert to a torch tensor
        return to_tensor(image)

    def load_labels(self, token):

        # Load label image as a torch tensor
        label_path = os.path.join(self.label_dir, token + ".png")
        encoded_labels = to_tensor(Image.open(label_path)).long()

        # Decode to binary labels
        num_class = len(nusc_utils.NUSCENES_CLASS_NAMES)
        labels = nusc_utils.decode_binary_labels(encoded_labels, num_class + 1)
        labels, mask = labels[:-1], ~labels[-1]

        return labels, mask


if __name__ == "__main__":

    dataset = NuSceneDataset(data_root="data", label_dir="labels")

    print(dataset.get_tokens())
    image = dataset.load_image(dataset.tokens[0])
    print(image.shape)
    labels, mask = dataset.load_labels(dataset.tokens[0])
    print(labels.shape)
    print(mask.shape)

    train_loader = DataLoader(dataset,
                            batch_size=2,
                            num_workers=2,
                            pin_memory=True,
                            shuffle=True)
    # print(len(dataset))
    # data_loader = DataLoader(dataset, batch_size=2)
    # image, labels, mask = next(iter(data_loader))
    # print(image.shape)
    # print(labels.shape)
    # print(mas.shape)
    # sample = [ x[0] for x in iter(data_loader).next() ]

    image, labels, mask = dataset[0]
    print(labels)