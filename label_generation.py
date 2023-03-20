import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from nuscenes import NuScenes
import nuscenes_utilities as nusc_utils
from configs.config_utilities import load_config


def process_scene(nuscenes, map_data, scene, config):

    # Get the map corresponding to the current sample data
    log = nuscenes.get("log", scene["log_token"])
    scene_map_data = map_data[log["location"]]

    # Iterate over samples
    first_sample_token = scene["first_sample_token"]

    for sample in nusc_utils.iterate_samples(nuscenes, first_sample_token):
        process_sample(nuscenes, scene_map_data, sample, config)


def process_sample(nuscenes, map_data, sample, config):

    # Load the lidar point cloud associated with this sample
    lidar_data = nuscenes.get("sample_data", sample["data"]["LIDAR_TOP"])
    lidar_pcl = nusc_utils.load_point_cloud(nuscenes, lidar_data)

    # Transform points into world coordinate system
    lidar_transform = nusc_utils.get_sensor_transform(nuscenes, lidar_data)
    lidar_pcl = nusc_utils.transform(lidar_transform, lidar_pcl)

    sample_data = nuscenes.get("sample_data", sample["data"]["CAM_FRONT"])
    process_sample_data(nuscenes, map_data, sample_data, lidar_pcl, config)


def process_sample_data(nuscenes, map_data, sample_data, lidar, config):

    # Render static road geometry masks
    map_masks = nusc_utils.get_map_masks(
        nuscenes,
        map_data,
        sample_data,
        config.map_extents,
        config.map_resolution,
    )

    # Render dynamic object masks
    obj_masks = nusc_utils.get_object_masks(
        nuscenes,
        sample_data,
        config.map_extents,
        config.map_resolution,
    )

    masks = np.concatenate([map_masks, obj_masks], axis=0)

    # Ignore regions of the BEV which are outside the image
    sensor = nuscenes.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    intrinsics = np.array(sensor["camera_intrinsic"])
    masks[-1] |= ~nusc_utils.get_visible_mask(
        intrinsics,
        sample_data["width"],
        config.map_extents,
        config.map_resolution,
    )
    visible_mask = masks[-1]
    # Transform lidar points into camera coordinates
    cam_transform = nusc_utils.get_sensor_transform(nuscenes, sample_data)
    cam_points = nusc_utils.transform(np.linalg.inv(cam_transform), lidar)
    masks[-1] |= nusc_utils.get_occlusion_mask(
        cam_points,
        config.map_extents,
        config.map_resolution,
    )
    occlusion_mask = masks[-1]

    # Encode masks as integer bitmask
    labels = nusc_utils.encode_binary_labels(masks)

    # Save outputs to disk
    output_path = os.path.join(
        os.path.expandvars(config.label_root),
        sample_data["token"] + ".png",
    )
    Image.fromarray(labels.astype(np.int32), mode="I").save(output_path)


if __name__ == "__main__":

    config = load_config("configs\configs.yml")

    dataroot = os.path.join(os.getcwd(), config.nuscenes_dataroot)
    nuscenes = NuScenes(config.nuscenes_version, dataroot)

    map_data = nusc_utils.load_map_data(dataroot)

    output_root = os.path.expandvars(config.label_root)
    os.makedirs(output_root, exist_ok=True)

    for scene in tqdm(nuscenes.scene[:1]):
        process_scene(nuscenes, map_data, scene, config)
