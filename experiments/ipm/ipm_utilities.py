from typing import Tuple
import torch
import kornia

import matplotlib.pyplot as plt


def cal_bev_coord(
    pts_u: torch.Tensor,
    pts_v: torch.Tensor,
    intrinsics: torch.Tensor,
    map_extents: Tuple[float, float, float, float],
    map_resolution: float,
    car_height: float,
) -> torch.Tensor:
    """
    Calculate BEV coordinates.

    Parameters
    -----
    pts_u: tensor with shape (batch, 4)
    pts_v: tensor with shape (batch, 4)
    intrinsics : tensor with shape (batch, 3, 3)
    map_extents : tuple[float, float, float, float]
    map_resolution : float
    car_height : float

    Return
    -----
    xy_bev: tensor with shape (batch, 4, 2)
    """
    fx = intrinsics[:, 0, 0].unsqueeze(1)  # tensor with shape (batch, 1)
    fy = intrinsics[:, 1, 1].unsqueeze(1)  # tensor with shape (batch, 1)
    cx = intrinsics[:, 0, 2].unsqueeze(1)  # tensor with shape (batch, 1)
    cy = intrinsics[:, 1, 2].unsqueeze(1)  # tensor with shape (batch, 1)

    zc = (fy * (-car_height)) / (pts_v - cy)
    xc = zc * (pts_u - cx) / fx

    ### map_extents: [x_topleft, y_topleft, x_bottomright, y_bottomright]

    x_bev = (xc - map_extents[0]) / map_resolution  # tensor with shape (batch, 4)
    y_bev = (-zc - map_extents[1]) / map_resolution  # tensor with shape (batch, 4)

    xy_bev = torch.cat((x_bev.unsqueeze(2), y_bev.unsqueeze(2)), dim=2)

    return xy_bev


def ipm_transform(
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    map_extents: Tuple[float, float, float, float],
    map_resolution: float,
    car_height: float,
) -> torch.Tensor:
    """
    Inverse Perspective Mapping.

    Parameters
    -----
    images : tensor with shape (batch, channel, height, width)
    intrinsics : tensor with shape (batch, 3, 3)
    map_extents : tuple[float, float, float, float]
    map_resolution : float
    car_height : float

    Return
    -----
    bev : tensor with shape (batch, channel, height, width)
    """
    batch_size, _, height, width = images.shape

    pts1 = (
        torch.tensor(
            [
                (0, 0),
                (0, width - 1),
                (height - 1, 0),
                (height - 1, width - 1),
            ]
        )
        .float()
        .repeat(batch_size, 1, 1)
    )

    pts2 = cal_bev_coord(
        pts1[:, :, 0],
        pts1[:, :, 1],
        intrinsics,
        map_extents,
        map_resolution,
        car_height,
    )

    bev_width = round((map_extents[2] - map_extents[0]) / map_resolution)
    bev_height = round((map_extents[3] - map_extents[1]) / map_resolution)

    M_ipm = kornia.geometry.transform.get_perspective_transform(pts1, pts2)
    bev = kornia.geometry.transform.warp_perspective(
        images, M_ipm, (bev_height, bev_width)
    )

    return bev


if __name__ == "__main__":
    import sys
    import os

    sys.path.append("../..")
    from dataset import NuScenesDataset
    from configs.config_utilities import load_config
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    os.chdir("../..")

    config = load_config()

    dataset = NuScenesDataset(
        nuscenes_dir=config.nuscenes_dir,
        nuscenes_version=config.nuscenes_version,
        image_size=(200, 112),
        label_dir=config.label_dir,
        sample_tokens=["452cb8aa72de4124907764018407b8d8"],
    )

    def image_to_bev(batch):
        images, labels, masks, calibs = zip(*batch)

        bev_images = ipm_transform(
            torch.stack(images),
            torch.stack(calibs),
            config.map_extents,
            config.map_resolution,
            car_height=1.562,
        )

        return images, bev_images, labels, masks, calibs

    dataset_loader = DataLoader(
        dataset,
        batch_size=8,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
        collate_fn=image_to_bev,
    )

    images, bev_images, labels, masks, calibs = next(iter(dataset_loader))

    # bev_images = ipm_transform(
    #     images, calibs, config.map_extents, config.map_resolution, car_height=1.562
    # )

    # print(images.shape)

    fig, axes = plt.subplots(1, 5, dpi=300)

    axes[0].imshow(images[0].permute(1, 2, 0))
    axes[0].axis("off")

    axes[1].imshow(bev_images[0].permute(1, 2, 0))
    axes[1].axis("off")

    axes[2].imshow(labels[0][4], cmap="gray")
    axes[2].axis("off")

    axes[3].imshow(masks[0], cmap="gray")
    axes[3].axis("off")

    axes[4].imshow(masks[0] & labels[0][4], cmap="gray")
    axes[4].axis("off")

    plt.show()
