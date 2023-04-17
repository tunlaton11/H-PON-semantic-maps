import os
import numpy as np
from shapely.strtree import STRtree
from collections import OrderedDict
from pyquaternion import Quaternion
from shapely import geometry, affinity
import cv2
import torch

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.eval.detection.utils import category_to_detection_name


NUSCENES_CLASS_NAMES = [
    "drivable_area",
    "ped_crossing",
    "walkway",
    "carpark",
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "barrier",
]

STATIC_CLASSES = [
    "drivable_area",
    "ped_crossing",
    "walkway",
    "carpark_area",
]

LOCATIONS = [
    "boston-seaport",
    "singapore-onenorth",
    "singapore-queenstown",
    "singapore-hollandvillage",
]


def load_map_data(dataroot: str):
    map_data = {
        location: load_location_map_data(
            dataroot,
            location,
        )
        for location in LOCATIONS
    }
    return map_data


def load_location_map_data(dataroot: str, location: str):
    # Load the NuScenes map object
    nusc_map = NuScenesMap(dataroot, location)

    map_data = OrderedDict()

    for layer in STATIC_CLASSES:
        # Retrieve all data associated with the current layer
        records = getattr(nusc_map, layer)
        polygons = list()

        # Drivable area records can contain multiple polygons
        if layer == "drivable_area":
            for record in records:
                # Convert each entry in the record into a shapely object
                for token in record["polygon_tokens"]:
                    poly = nusc_map.extract_polygon(token)
                    if poly.is_valid:
                        polygons.append(poly)
        else:
            for record in records:
                # Convert each entry in the record into a shapely object
                poly = nusc_map.extract_polygon(record["polygon_token"])
                if poly.is_valid:
                    polygons.append(poly)

        # Store as an R-Tree for fast intersection queries
        map_data[layer] = STRtree(polygons)

    return map_data


def iterate_samples(nuscenes, start_token: str):
    sample_token = start_token
    while sample_token != "":
        sample = nuscenes.get("sample", sample_token)
        yield sample
        sample_token = sample["next"]


# -- utility functions for process_sample -- #


def load_point_cloud(nuscenes, sample_data):
    # Load point cloud
    lidar_path = os.path.join(nuscenes.dataroot, sample_data["filename"])
    pcl = LidarPointCloud.from_file(lidar_path)
    return pcl.points[:3, :].T


def get_sensor_transform(nuscenes, sample_data):
    # Load sensor transform data
    sensor = nuscenes.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    sensor_tfm = make_transform_matrix(sensor)

    # Load ego pose data
    pose = nuscenes.get("ego_pose", sample_data["ego_pose_token"])
    pose_tfm = make_transform_matrix(pose)

    return np.dot(pose_tfm, sensor_tfm)


def make_transform_matrix(record):
    """
    Create a 4x4 transform matrix from a calibrated_sensor or ego_pose record
    """
    transform = np.eye(4)
    transform[:3, :3] = Quaternion(record["rotation"]).rotation_matrix
    transform[:3, 3] = np.array(record["translation"])
    return transform


def transform(matrix, vectors):
    vectors = np.dot(matrix[:-1, :-1], vectors.T)
    vectors = vectors.T + matrix[:-1, -1]
    return vectors


# -- utility functions for process_sample_data -- #


def transform_polygon(polygon, affine):
    """
    Transform a 2D polygon
    """
    a, b, tx, c, d, ty = affine.flatten()[:6]
    return affinity.affine_transform(polygon, [a, b, c, d, tx, ty])


def render_polygon(mask, polygon, extents, resolution, value=1):
    if len(polygon) == 0:
        return
    polygon = (polygon - np.array(extents[:2])) / resolution
    polygon = np.ascontiguousarray(polygon).round().astype(np.int32)
    cv2.fillConvexPoly(mask, polygon, value)


def get_map_masks(nuscenes, map_data, sample_data, extents, resolution):
    # Render each layer sequentially
    layers = [
        get_layer_mask(nuscenes, polys, sample_data, extents, resolution)
        for layer, polys in map_data.items()
    ]

    return np.stack(layers, axis=0)


def get_layer_mask(nuscenes, polygons, sample_data, extents, resolution):
    # Get the 2D affine transform from bev coords to map coords
    tfm = get_sensor_transform(nuscenes, sample_data)[[0, 1, 3]][:, [0, 2, 3]]
    inv_tfm = np.linalg.inv(tfm)

    # Create a patch representing the birds-eye-view region in map coordinates
    map_patch = geometry.box(*extents)
    map_patch = transform_polygon(map_patch, tfm)

    # Initialise the map mask
    x1, z1, x2, z2 = extents
    mask = np.zeros(
        (int((z2 - z1) / resolution), int((x2 - x1) / resolution)), dtype=np.uint8
    )

    # Find all polygons which intersect with the area of interest
    for polygon in polygons.query(map_patch):
        polygon = polygons.geometries.take(polygon)
        polygon = polygon.intersection(map_patch)

        # Transform into map coordinates
        polygon = transform_polygon(polygon, inv_tfm)

        # Render the polygon to the mask
        render_shapely_polygon(mask, polygon, extents, resolution)

    return mask.astype(bool)


DETECTION_NAMES = [
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "barrier",
]


def get_object_masks(nuscenes, sample_data, extents, resolution):
    # Initialize object masks
    nclass = len(DETECTION_NAMES) + 1
    grid_width = int((extents[2] - extents[0]) / resolution)
    grid_height = int((extents[3] - extents[1]) / resolution)
    masks = np.zeros((nclass, grid_height, grid_width), dtype=np.uint8)

    # Get the 2D affine transform from bev coords to map coords
    tfm = get_sensor_transform(nuscenes, sample_data)[[0, 1, 3]][:, [0, 2, 3]]
    inv_tfm = np.linalg.inv(tfm)

    for box in nuscenes.get_boxes(sample_data["token"]):
        # Get the index of the class
        det_name = category_to_detection_name(box.name)
        if det_name not in DETECTION_NAMES:
            class_id = -1
        else:
            class_id = DETECTION_NAMES.index(det_name)

        # Get bounding box coordinates in the grid coordinate frame
        bbox = box.bottom_corners()[:2]
        local_bbox = np.dot(inv_tfm[:2, :2], bbox).T + inv_tfm[:2, 2]

        # Render the rotated bounding box to the mask
        render_polygon(masks[class_id], local_bbox, extents, resolution)

    return masks.astype(bool)


def get_visible_mask(instrinsics, image_width, extents, resolution):
    # Get calibration parameters
    fu, cu = instrinsics[0, 0], instrinsics[0, 2]

    # Construct a grid of image coordinates
    x1, z1, x2, z2 = extents
    x, z = np.arange(x1, x2, resolution), np.arange(z1, z2, resolution)
    ucoords = x / z[:, None] * fu + cu

    # Return all points which lie within the camera bounds
    return (ucoords >= 0) & (ucoords < image_width)


def render_shapely_polygon(mask, polygon, extents, resolution):
    if polygon.geom_type == "Polygon":
        # Render exteriors
        render_polygon(mask, polygon.exterior.coords, extents, resolution, 1)

        # Render interiors
        for hole in polygon.interiors:
            render_polygon(mask, hole.coords, extents, resolution, 0)

    # Handle the case of compound shapes
    else:
        for poly in list(polygon.geoms):
            render_shapely_polygon(mask, poly, extents, resolution)


def get_occlusion_mask(points, extents, resolution):
    x1, z1, x2, z2 = extents

    # A 'ray' is defined by the ratio between x and z coordinates
    ray_width = resolution / z2
    ray_offset = x1 / ray_width
    max_rays = int((x2 - x1) / ray_width)

    # Group LiDAR points into bins
    rayid = np.round(points[:, 0] / points[:, 2] / ray_width - ray_offset)
    depth = points[:, 2]

    # Ignore rays which do not correspond to any grid cells in the BEV
    valid = (rayid > 0) & (rayid < max_rays) & (depth > 0)
    rayid = rayid[valid]
    depth = depth[valid]

    # Find the LiDAR point with maximum depth within each bin
    max_depth = np.zeros((max_rays,))
    np.maximum.at(max_depth, rayid.astype(np.int32), depth)

    # For each bev grid point, sample the max depth along the corresponding ray
    x = np.arange(x1, x2, resolution)
    z = np.arange(z1, z2, resolution)[:, None]
    grid_rayid = np.round(x / z / ray_width - ray_offset).astype(np.int32)
    grid_max_depth = max_depth[grid_rayid]

    # A grid position is considered occluded if the there are no LiDAR points
    # passing through it
    occluded = grid_max_depth < z
    return occluded


def encode_binary_labels(masks):
    bits = np.power(2, np.arange(len(masks), dtype=np.int32))
    # print((masks.astype(np.int32) * bits.reshape(-1, 1, 1)).sum(0))
    return (masks.astype(np.int32) * bits.reshape(-1, 1, 1)).sum(0)


def decode_binary_labels_old(labels, nclass):
    bits = torch.pow(2, torch.arange(nclass))
    return (labels & bits.view(-1, 1, 1)) > 0


def decode_binary_labels(
    encoded_labels: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    bits = 2 ** np.arange(n_classes, dtype=np.int32)
    bits = bits.reshape(-1, 1, 1)
    encoded_labels = encoded_labels.astype(np.int32)
    return ((encoded_labels & bits) > 0).astype(int)


def flatten_labels(
    labels: np.ndarray,
    mask: np.ndarray = None,
) -> np.ndarray:
    flattened_label = np.zeros_like(labels[0])
    for i, label in enumerate(labels):
        label = label * (i + 1)
        flattened_label = np.maximum(flattened_label, label)
    if mask is not None:
        flattened_label[~mask] = 0
    return flattened_label
