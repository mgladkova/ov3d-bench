"""{name} -> Omni3D-format converter for OV3D-Bench.

Authored by Neehar Peri for the omni3d-xl pipeline and included here with his
permission as a co-author. Adapted only in its imports: the hardcoded
`sys.path.append` for the av2 devkit is gone, the unused `vis` import is dropped,
and the vocabulary and geometry helpers now come from the installed package.

Requires the source dataset downloaded under its own terms. See docs/DATA.md.
"""


import os
import shutil
from collections import OrderedDict
from os import path as osp
from typing import List, Tuple, Union

import numpy as np
import pickle 
from tqdm import tqdm 
from pathlib import Path
import math
import copy
import torch 
import cv2

from av2.datasets.sensor.splits import TRAIN, VAL
from av2.utils.io import read_feather, read_city_SE3_ego
from av2.geometry.geometry import quat_to_mat
from av2.geometry.se3 import SE3
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.structures.cuboid import Cuboid, CuboidList
from shapely.geometry import MultiPoint, box
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from ov3d_bench.omni3d import geometry as utils
from ov3d_bench.omni3d.geometry import array_converter
from _vocab import CLASS_MAPPING, CLASSES

TRAIN_SAMPLE_RATE = 10
VAL_SAMPLE_RATE = 10

@array_converter(apply_to=('points_3d', 'proj_mat'))
def points_cam2img(points_3d, proj_mat, with_depth=False):
    """Project points in camera coordinates to image coordinates.

    Args:
        points_3d (torch.Tensor | np.ndarray): Points in shape (N, 3)
        proj_mat (torch.Tensor | np.ndarray):
            Transformation matrix between coordinates.
        with_depth (bool, optional): Whether to keep depth in the output.
            Defaults to False.

    Returns:
        (torch.Tensor | np.ndarray): Points in image coordinates,
            with shape [N, 2] if `with_depth=False`, else [N, 3].
    """
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1

    assert len(proj_mat.shape) == 2, 'The dimension of the projection'\
        f' matrix should be 2 instead of {len(proj_mat.shape)}.'
    d1, d2 = proj_mat.shape[:2]
    assert (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or (
        d1 == 4 and d2 == 4), 'The shape of the projection matrix'\
        f' ({d1}*{d2}) is not supported.'
    if d1 == 3:
        proj_mat_expanded = torch.eye(
            4, device=proj_mat.device, dtype=proj_mat.dtype)
        proj_mat_expanded[:d1, :d2] = proj_mat
        proj_mat = proj_mat_expanded

    # previous implementation use new_zeros, new_one yields better results
    points_4 = torch.cat([points_3d, points_3d.new_ones(points_shape)], dim=-1)

    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

    if with_depth:
        point_2d_res = torch.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)

    return point_2d_res

def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.
    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False
    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points

def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.
    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.
    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str, cat_name: str) -> OrderedDict:
    """Generate one 2D annotation record given various information on top of
    the 2D bounding box coordinates.
    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.
    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): file name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    coco_rec = dict()

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = CLASSES.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2, y2]
    coco_rec['iscrowd'] = 0

    return coco_rec

def yaw_to_quaternion3d(yaw: float) -> np.ndarray:
    """Convert a rotation angle in the xy plane (i.e. about the z axis) to a quaternion.
    Args:
        yaw: angle to rotate about the z-axis, representing an Euler angle, in radians
    Returns:
        array w/ quaternion coefficients (qw,qx,qy,qz) in scalar-first order, per Argoverse convention.
    """
    qx, qy, qz, qw = Rotation.from_euler(seq="z", angles=yaw, degrees=False).as_quat()
    return np.array([qw, qx, qy, qz])
   
def yaw_to_mat(yaw: float) -> np.ndarray:
    """Convert a rotation angle in the xy plane (i.e. about the z axis) to a quaternion.
    Args:
        yaw: angle to rotate about the z-axis, representing an Euler angle, in radians
    Returns:
        array w/ quaternion coefficients (qw,qx,qy,qz) in scalar-first order, per Argoverse convention.
    """
    return Rotation.from_euler(seq="y", angles=yaw, degrees=False).as_matrix()

def generate_info(filename, log_id, annotations, class_names):
    timestamp_ns = int(filename.split(".")[0])

    if annotations is None:
        gt_bboxes_3d = []
        gt_labels = []
        gt_names = [] 
        gt_num_pts = []
        
    else:
        curr_annotations = annotations[annotations["timestamp_ns"] == timestamp_ns]
        curr_annotations = curr_annotations[curr_annotations["num_interior_pts"] > 5]

        gt_bboxes_3d = []
        gt_labels = []
        gt_names = [] 
        gt_num_pts = []

        for annotation in curr_annotations.iterrows():
            class_name = CLASS_MAPPING[annotation[1]["category"]]

            if class_name not in CLASSES:
                continue 

            num_interior_pts = annotation[1]["num_interior_pts"]

            gt_labels.append(CLASSES.index(class_name))
            gt_names.append(class_name)
            gt_num_pts.append(num_interior_pts)

            translation = np.array([annotation[1]["tx_m"], annotation[1]["ty_m"], annotation[1]["tz_m"]])
            lwh = np.array([annotation[1]["length_m"], annotation[1]["width_m"], annotation[1]["height_m"]])
            rotation = quat_to_mat(np.array([annotation[1]["qw"], annotation[1]["qx"], annotation[1]["qy"], annotation[1]["qz"]]))
            ego_SE3_object = SE3(rotation=rotation, translation=translation)

            rot = ego_SE3_object.rotation
            lwh = lwh.tolist()
            center = translation.tolist()
            center[2] = center[2] - lwh[2] / 2
            yaw = math.atan2(rot[1, 0], rot[0, 0])

            gt_bboxes_3d.append([*center, *lwh, yaw])
    
    info = {
        'log_id': log_id,
        'timestamp': timestamp_ns,
        'gt_bboxes' : gt_bboxes_3d,
        'gt_labels' : gt_labels,
        'gt_names' : gt_names, 
        'gt_num_pts' : gt_num_pts,
    }

    return info 

def get_2d_boxes(info, cam_img, cam_model, timestamp_city_SE3_ego_dict, width, height, mono3d=True):
    timestamp = info["timestamp"]
    cam_timestamp = int(cam_img.split("/")[-1].split(".")[0])

    city_SE3_ego_reference = timestamp_city_SE3_ego_dict[timestamp]
    city_SE3_ego_cam_t = timestamp_city_SE3_ego_dict[cam_timestamp]
        
    log_id = info["log_id"]
    ego_SE3_cam = cam_model.ego_SE3_cam
    camera_intrinsic = cam_model.intrinsics.K
    
    coco_infos = []
    for name, bbox, num_pts in zip(info["gt_names"], info["gt_bboxes"], info["gt_num_pts"]):
        quat = yaw_to_quaternion3d(bbox[-1]).tolist()
        cuboid_ego = Cuboid.from_numpy(np.array(bbox[:-1] + quat), name, timestamp)
        
        cuboid_ego_av2 = copy.deepcopy(cuboid_ego)
        cuboid_ego_av2.dst_SE3_object.translation[2] = cuboid_ego_av2.dst_SE3_object.translation[2] + cuboid_ego_av2.height_m / 2
        
        reference_SE3_ego_cam_t = city_SE3_ego_reference.inverse().compose(city_SE3_ego_cam_t)
        cuboid_ego_av2 = cuboid_ego_av2.transform(reference_SE3_ego_cam_t.inverse())
 
        cuboid_cam = cuboid_ego_av2.transform(ego_SE3_cam.inverse())
        cam_box = CuboidList([cuboid_cam])

        cuboids_vertices_cam = cam_box.vertices_m
        N, V, D = cuboids_vertices_cam.shape

        # Collapse first dimension to allow for vectorization.
        cuboids_vertices_cam = cuboids_vertices_cam.reshape(-1, D)
        _, _, is_valid = cam_model.project_cam_to_img(cuboids_vertices_cam)

        num_valid = np.sum(is_valid)
        if num_valid > 0:
            corner_coords = view_points(cuboid_cam.vertices_m.T, camera_intrinsic, True).T[:, :2].tolist()

            # Keep only corners that fall within the image.
            final_coords = post_process_coords(corner_coords, (width, height))

            # Skip if the convex hull of the re-projected corners
            # does not intersect the image canvas.
            if final_coords is None:
                continue
            else:
                min_x, min_y, max_x, max_y = final_coords

            repro_rec = generate_record(min_x, min_y, max_x, max_y, log_id, cam_img, name)

            if mono3d and (repro_rec is not None):
                rot = cuboid_ego.dst_SE3_object.rotation
                #size = [cuboid_ego.width_m, cuboid_ego.height_m, cuboid_ego.length_m ]
                size = [cuboid_ego.height_m, cuboid_ego.width_m,  cuboid_ego.length_m ]

                center = cuboid_ego.dst_SE3_object.translation.tolist()
                yaw = math.atan2(rot[2,1], rot[2,1]) + cam_model.egovehicle_yaw_cam_rad
                #yaw = math.acos(rot[2,2]) - cam_model.egovehicle_yaw_cam_rad
                #yaw = -math.atan2(rot[0,2], rot[1,2]) - cam_model.egovehicle_yaw_cam_rad

                repro_rec['bbox_cam3d'] = [*center, *size, yaw]

                center2d = points_cam2img(cuboid_cam.dst_SE3_object.translation.tolist(), camera_intrinsic, with_depth=True)

                repro_rec['center2d'] = center2d.squeeze().tolist()
                # normalized center2D + depth
                # if samples with depth < 0 will be removed
                if repro_rec['center2d'][2] <= 0:
                    continue

                repro_rec['attribute_name'] = "None"
                repro_rec['attribute_id'] = 0
                repro_rec['gt_num_pts'] = num_pts

                center_cam = cuboid_cam.dst_SE3_object.translation.tolist() 
                box3d = center_cam + size
                R = cuboid_cam.dst_SE3_object.rotation
          
                box3d_corners = cuboids_vertices_cam[[6, 2, 3, 7, 5, 1, 0, 4]]
                truncation = utils.estimate_truncation(camera_intrinsic, box3d, R, width, height)
                bbox, behind_camera, fully_behind = utils.convert_3d_box_to_2d(camera_intrinsic, box3d, R, width, height)

                omni3d = {"bbox2D_tight" : [-1, -1, -1, -1],
                    "bbox2D_proj" : bbox,
                    "bbox2D_trunc" : repro_rec["bbox"],
                    "bbox3D_cam" : box3d_corners,
                    "center_cam" : center_cam,
                    "dimensions" : size,
                    "R_cam" : R,
                    "behind_camera" : behind_camera.item(),
                    "visibility" : -1,
                    "truncation" : truncation,
                    "segmentation_pts" : -1,
                    "lidar_pts" : num_pts,
                    "depth_error" : -1,
                    "valid3D": not bool(fully_behind or truncation >= 1.0),
                    "category_name" : name,
                    "category_id" : CLASSES.index(name)}

            coco_infos.append(omni3d)
            
        
    return coco_infos

def create_infos(root_path, out_path, stats):
    train_infos = {"info" : {"source" : "AV2", 
                                "split" : "train", 
                                "name" : "AV2-train", 
                                "id": stats["n_datasets"], 
                                "version" : "1.0"},
                    "images" : [],
                    "annotations" : [],
                    "categories" : [{"id" : CLASSES.index(name), "name" : name} for name in CLASSES]
                }
    val_infos = {"info" : {"source" : "AV2", 
                                   "split" : "val", 
                                   "name" : "AV2-val", 
                                   "id": stats["n_datasets"], 
                                   "version" : "1.0"},
                        "images" : [],
                        "annotations" : [],
                        "categories" : [{"id" : CLASSES.index(name), "name" : name} for name in CLASSES]
                    }
    
    for log_id in tqdm(TRAIN):
        split = "train"

        log_dir = "{root_path}/{split}/{log_id}".format(root_path=root_path, split=split, log_id=log_id)
        lidar_paths = "{log_dir}/sensors/lidar".format(log_dir=log_dir)
        annotations_path = "{log_dir}/annotations.feather".format(log_dir=log_dir)
        annotations = read_feather(Path(annotations_path))

        for i, filename in enumerate(sorted(os.listdir(lidar_paths))):
            if i % TRAIN_SAMPLE_RATE != 0:
                continue 

            info = generate_info(filename, log_id, annotations, CLASSES)
            
            camera_types = [
                    'ring_front_center',
                ]
            
            log_id = info["log_id"]
            timestamp = info["timestamp"]

            log_dir = Path("{}/{}/{}".format(root_path, split, log_id))
            timestamp_city_SE3_ego_dict = read_city_SE3_ego(Path(log_dir))

            cam_imgs, cam_models = {}, {}
            for cam_name in camera_types:
                cam_models[cam_name] = PinholeCamera.from_feather(log_dir, cam_name)

                cam_path = root_path + "{}/{}/sensors/cameras/{}/".format(split, log_id, cam_name)
                closest_dst = np.inf
                closest_img = None
                for filename in os.listdir(cam_path):
                    img_timestamp = int(filename.split(".")[0])
                    delta = abs(timestamp - img_timestamp)

                    if delta < closest_dst:
                        closest_img = cam_path + filename
                        closest_dst = delta
                        
                cam_imgs[cam_name] = closest_img

            for cam_name in camera_types:
                cam_img = cam_imgs[cam_name]
                cam_model = cam_models[cam_name]
                
                height, width = cam_model.intrinsics.height_px, cam_model.intrinsics.width_px

                image_info = {"width" : width,
                            "height" : height,
                            "file_path" : "AV2/" + cam_imgs[cam_name].replace(root_path, ""),
                            "K" : cam_model.intrinsics.K.tolist(),
                            "src_90_rotate" : 0,
                            "src_flagged" : False, 
                            "incomplete" : False,
                            "id": stats['n_ims'], 
                            "dataset_id" : stats["n_datasets"],
                            }
                
                out_dir = os.path.dirname(out_path + cam_imgs[cam_name].replace(root_path, ""))
                os.makedirs(out_dir, exist_ok=True)
                
                if not os.path.isfile(out_path + cam_imgs[cam_name].replace(root_path, "")):
                    shutil.copyfile(root_path + cam_imgs[cam_name].replace(root_path, ""), out_path + cam_imgs[cam_name].replace(root_path, ""))
                
                coco_infos = get_2d_boxes(info, cam_img, cam_model, timestamp_city_SE3_ego_dict, width, height, mono3d=True)
                                
                for anno in coco_infos: 
                    anno_info = {"bbox2D_tight" : anno["bbox2D_tight"],
                        "bbox2D_proj" : anno["bbox2D_proj"].tolist(),
                        "bbox2D_trunc" : anno["bbox2D_proj"].tolist(),
                        "valid3D" : anno["valid3D"],
                        "bbox3D_cam" : anno["bbox3D_cam"].tolist(),
                        "center_cam" : anno["center_cam"],
                        "dimensions" : anno["dimensions"],
                        "R_cam" : anno["R_cam"].tolist(),
                        "behind_camera" : anno["behind_camera"],
                        "visibility" : anno["visibility"],
                        "truncation" : anno["truncation"],
                        "segmentation_pts" : anno["segmentation_pts"],
                        "lidar_pts" : anno["lidar_pts"],
                        "depth_error" : anno["depth_error"],
                        "category_id" : anno["category_id"],
                        "category_name" : anno["category_name"],
                        "image_id" : stats['n_ims'],
                        "id" : stats['n_anns'],
                        "dataset_id" : stats['n_datasets']
                    }
                
                    train_infos["annotations"].append(anno_info)                    
                    stats['n_anns'] += 1
                
                train_infos["images"].append(image_info)
                stats['n_ims'] += 1
    
    for log_id in tqdm(VAL):
        split = "val"

        log_dir = "{root_path}/{split}/{log_id}".format(root_path=root_path, split=split, log_id=log_id)
        lidar_paths = "{log_dir}/sensors/lidar".format(log_dir=log_dir)
        annotations_path = "{log_dir}/annotations.feather".format(log_dir=log_dir)
        annotations = read_feather(Path(annotations_path))

        for i, filename in enumerate(sorted(os.listdir(lidar_paths))):
            if i % VAL_SAMPLE_RATE != 0:
                continue 

            info = generate_info(filename, log_id, annotations, CLASSES)
            
            camera_types = [
                    'ring_front_center',
                ]
            
            log_id = info["log_id"]
            timestamp = info["timestamp"]

            log_dir = Path("{}/{}/{}".format(root_path, split, log_id))
            timestamp_city_SE3_ego_dict = read_city_SE3_ego(Path(log_dir))

            cam_imgs, cam_models = {}, {}
            for cam_name in camera_types:
                cam_models[cam_name] = PinholeCamera.from_feather(log_dir, cam_name)

                cam_path = root_path + "{}/{}/sensors/cameras/{}/".format(split, log_id, cam_name)
                closest_dst = np.inf
                closest_img = None
                for filename in os.listdir(cam_path):
                    img_timestamp = int(filename.split(".")[0])
                    delta = abs(timestamp - img_timestamp)

                    if delta < closest_dst:
                        closest_img = cam_path + filename
                        closest_dst = delta
                        
                cam_imgs[cam_name] = closest_img

            for cam_name in camera_types:
                cam_img = cam_imgs[cam_name]
                cam_model = cam_models[cam_name]
                
                height, width = cam_model.intrinsics.height_px, cam_model.intrinsics.width_px

                image_info = {"width" : width,
                            "height" : height,
                            "file_path" : "AV2/" + cam_imgs[cam_name].replace(root_path, ""),
                            "K" : cam_model.intrinsics.K.tolist(),
                            "src_90_rotate" : 0,
                            "src_flagged" : False, 
                            "incomplete" : False,
                            "id": stats['n_ims'], 
                            "dataset_id" : stats["n_datasets"],
                            }
                
                out_dir = os.path.dirname(out_path + cam_imgs[cam_name].replace(root_path, ""))
                os.makedirs(out_dir, exist_ok=True)
                
                if not os.path.isfile(out_path + cam_imgs[cam_name].replace(root_path, "")):
                    shutil.copyfile(root_path + cam_imgs[cam_name].replace(root_path, ""), out_path + cam_imgs[cam_name].replace(root_path, ""))
                
                coco_infos = get_2d_boxes(info, cam_img, cam_model, timestamp_city_SE3_ego_dict, width, height, mono3d=True)
                
                for anno in coco_infos: 
                    anno_info = {"bbox2D_tight" : anno["bbox2D_tight"],
                        "bbox2D_proj" : anno["bbox2D_proj"].tolist(),
                        "bbox2D_trunc" : anno["bbox2D_proj"].tolist(),
                        "valid3D" : anno["valid3D"],
                        "bbox3D_cam" : anno["bbox3D_cam"].tolist(),
                        "center_cam" : anno["center_cam"],
                        "dimensions" : anno["dimensions"],
                        "R_cam" : anno["R_cam"].tolist(),
                        "behind_camera" : anno["behind_camera"],
                        "visibility" : anno["visibility"],
                        "truncation" : anno["truncation"],
                        "segmentation_pts" : anno["segmentation_pts"],
                        "lidar_pts" : anno["lidar_pts"],
                        "depth_error" : anno["depth_error"],
                        "category_id" : anno["category_id"],
                        "category_name" : anno["category_name"],
                        "image_id" : stats['n_ims'],
                        "id" : stats['n_anns'],
                        "dataset_id" : stats['n_datasets']
                    }

                    val_infos["annotations"].append(anno_info)                    
                    stats['n_anns'] += 1
                            
                val_infos["images"].append(image_info)
                stats['n_ims'] += 1
    
    print("Done Processing AV2")
    return train_infos, val_infos, stats


