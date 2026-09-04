"""{name} -> Omni3D-format converter for OV3D-Bench.

Authored by Neehar Peri for the omni3d-xl pipeline and included here with his
permission as a co-author. Adapted only in its imports: the hardcoded
`sys.path.append` for the av2 devkit is gone, the unused `vis` import is dropped,
and the vocabulary and geometry helpers now come from the installed package.

Requires the source dataset downloaded under its own terms. See docs/DATA.md.
"""

import json
import numpy as np
import cv2
import os
from itertools import product
import csv

from tqdm import tqdm
from _vocab import CLASS_MAPPING, CLASSES
from ov3d_bench.omni3d import geometry as utils 
import shutil

def represents_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[int(row[label_from])] = row[label_to]
    return mapping

def get_bbox_corners(centroid, axes_lengths):
    signs = np.array(list(product([-0.5, 0.5], repeat=3)))
    corners = signs * axes_lengths
    return corners + centroid

def transform_to_camera(corners_world, cam_pose_inv):
    corners_hom = np.hstack([corners_world, np.ones((8, 1))])
    corners_cam = (cam_pose_inv @ corners_hom.T).T[:, :3]
    return corners_cam

def project_to_2d(corners_cam, intrinsics):
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    x, y, z = corners_cam[:, 0], corners_cam[:, 1], corners_cam[:, 2]

    if np.any(z <= 0):
        return None  # skip invalid projections

    u = (fx * x / z) + cx
    v = (fy * y / z) + cy
    return np.stack([u, v], axis=1).astype(int)

def create_infos(root_path, out_path, stats):
    train_infos = {"info" : {"source" : "ScanNet", 
                                "split" : "train", 
                                "name" : "ScanNet-train", 
                                "id": stats["n_datasets"], 
                                "version" : "1.0"},
                    "images" : [],
                    "annotations" : [],
                    "categories" : [{"id" : CLASSES.index(name), "name" : name} for name in CLASSES]
                }
    val_infos = {"info" : {"source" : "ScanNet", 
                                   "split" : "val", 
                                   "name" : "ScanNet-val", 
                                   "id": stats["n_datasets"], 
                                   "version" : "1.0"},
                        "images" : [],
                        "annotations" : [],
                        "categories" : [{"id" : CLASSES.index(name), "name" : name} for name in CLASSES]
                    }
    
    label_map = read_label_mapping(root_path + 'scannetv2-labels.combined.tsv', label_from='id', label_to='raw_category')

    train_scenes = set([path.split("_")[0] + "_" + path.split("_")[1] for path in os.listdir(root_path + "scannet_train_detection_data")])
    val_scenes = set([path.split("_")[0] + "_" + path.split("_")[1] for path in os.listdir(root_path + "scannet_val_detection_data")])

    for train_scene in tqdm(train_scenes):
        scene_path = root_path + "tasks/scannet_frames_25k/{}".format(train_scene)
        intrinsics_path = os.path.join(scene_path, "intrinsics_color.txt")
        bboxes_npy = root_path + "scannet_train_detection_data/{}_bbox.npy".format(train_scene)
        # ---------------------------------------

        meta_file = root_path + "scans/{}/{}.txt".format(train_scene, train_scene)
        lines = open(meta_file).readlines()
        for line in lines:
            if 'axisAlignment' in line:
                axis_align_matrix = [float(x) \
                    for x in line.rstrip().strip('axisAlignment = ').split(' ')]
                break
        axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
        intrinsics = np.loadtxt(intrinsics_path)
        K = intrinsics[:3, :3]
        
        frame_ids = [int(file.split("/")[-1].replace(".jpg", "")) for file in os.listdir(os.path.join(scene_path, f"color"))]

        for frame_id in frame_ids:  # e.g., frame-000000
            image_path = os.path.join(scene_path, f"color/{frame_id:06d}.jpg")
            pose_path = os.path.join(scene_path, f"pose/{frame_id:06d}.txt")

            out_dir = os.path.dirname(out_path + image_path.replace(root_path, ""))
            os.makedirs(out_dir, exist_ok=True)
            
            if not os.path.isfile(out_path + image_path.replace(root_path, "")):
                shutil.copyfile(root_path + image_path.replace(root_path, ""), out_path + image_path.replace(root_path, ""))
                
            # Load pose and invert
            cam_pose = np.loadtxt(pose_path)
            cam_pose_inv = np.linalg.inv(axis_align_matrix @ cam_pose)

            # Load image
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            image_info = {"width" : width,
                        "height" : height,
                        "file_path" : "ScanNet/" + image_path.replace(root_path, ""),
                        "K" : K.tolist(),
                        "src_90_rotate" : 0,
                        "src_flagged" : False, 
                        "incomplete" : False,
                        "id": stats['n_ims'], 
                        "dataset_id" : stats["n_datasets"],
                        }
                        
            # Load bounding boxes
            bboxes = np.load(bboxes_npy)

            for bbox in bboxes:
                centroid = [bbox[0], bbox[1], bbox[2]]
                dimensions = [bbox[4], bbox[5], bbox[3]] #4, 5, 3
                
                axis = [ax[1] for ax in sorted([(dimensions[0], "x"), (dimensions[1], "y"), (dimensions[2], "z")], key = lambda x : x[0])]

                class_id = int(bbox[6])
                class_name = label_map[class_id]
                
                centroid_cam = (cam_pose_inv @ np.array([centroid[0], centroid[1], centroid[2], 1]))[:3]
                corners_world = get_bbox_corners(centroid, dimensions)
                corners_cam = transform_to_camera(corners_world, cam_pose_inv)
            
                corners_2d = project_to_2d(corners_cam, intrinsics)
                
                if corners_2d is None:
                    continue 
                        
                centered = corners_cam - centroid
                cov = np.cov(centered.T)
                _, R = np.linalg.eigh(cov)
                R = np.stack([R[:, axis.index("x")], R[:, axis.index("z")], R[:, axis.index("y")]], axis=1) #x, z, y

                bbox3d = np.hstack([centroid_cam, dimensions]).tolist()
                verts2d, verts3d = utils.get_cuboid_verts(K, bbox3d, R)
                
                truncation = utils.estimate_truncation(K, bbox3d, R, width, height)
                bbox, behind_camera, fully_behind = utils.convert_3d_box_to_2d(K, bbox3d, R, width, height)

                anno_info = {"bbox2D_tight" : [-1, -1, -1, -1],
                                "bbox2D_proj" : bbox.tolist(),
                                "bbox2D_trunc" : bbox.tolist(),
                                "valid3D" : not bool(fully_behind or truncation >= 1.0),
                                "bbox3D_cam" : verts3d.tolist(),
                                "center_cam" : centroid_cam.tolist(),
                                "dimensions" : dimensions,
                                "R_cam" : R.tolist(),
                                "behind_camera" : behind_camera.item(),
                                "visibility" : -1,
                                "truncation" : truncation,
                                "segmentation_pts" : -1,
                                "lidar_pts" : -1,
                                "depth_error" : -1,
                                "category_id" : CLASSES.index(CLASS_MAPPING[class_name]),
                                "category_name" : CLASS_MAPPING[class_name],
                                "image_id" : stats['n_ims'],
                                "id" : stats['n_anns'],
                                "dataset_id" : stats['n_datasets']
                            }
                
                train_infos["annotations"].append(anno_info)                    
                stats['n_anns'] += 1
                
            train_infos["images"].append(image_info)
            stats['n_ims'] += 1
        
    for val_scene in tqdm(val_scenes):
        scene_path = root_path + "tasks/scannet_frames_25k/{}".format(val_scene)
        intrinsics_path = os.path.join(scene_path, "intrinsics_color.txt")
        bboxes_npy = root_path + "scannet_val_detection_data/{}_bbox.npy".format(val_scene)
        # ---------------------------------------

        meta_file = root_path + "scans/{}/{}.txt".format(val_scene, val_scene)
        lines = open(meta_file).readlines()
        for line in lines:
            if 'axisAlignment' in line:
                axis_align_matrix = [float(x) \
                    for x in line.rstrip().strip('axisAlignment = ').split(' ')]
                break
        axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
        intrinsics = np.loadtxt(intrinsics_path)
        K = intrinsics[:3, :3]
        
        frame_ids = [int(file.split("/")[-1].replace(".jpg", "")) for file in os.listdir(os.path.join(scene_path, f"color"))]

        for frame_id in frame_ids:
            image_path = os.path.join(scene_path, f"color/{frame_id:06d}.jpg")
            depth_path = os.path.join(scene_path, f"depth/{frame_id:06d}.png")
            pose_path = os.path.join(scene_path, f"pose/{frame_id:06d}.txt")

            out_dir = os.path.dirname(out_path + image_path.replace(root_path, ""))
            os.makedirs(out_dir, exist_ok=True)
            
            if not os.path.isfile(out_path + image_path.replace(root_path, "")):
                shutil.copyfile(root_path + image_path.replace(root_path, ""), out_path + image_path.replace(root_path, ""))
                            
            # Load pose and invert
            cam_pose = np.loadtxt(pose_path)
            cam_pose_inv = np.linalg.inv(axis_align_matrix @ cam_pose)

            # Load image
            image = cv2.imread(image_path)
            depth = cv2.imread(depth_path)
            height, width, _ = image.shape

            image_info = {"width" : width,
                        "height" : height,
                        "file_path" : "ScanNet/" + image_path.replace(root_path, ""),
                        "K" : K.tolist(),
                        "src_90_rotate" : 0,
                        "src_flagged" : False, 
                        "incomplete" : False,
                        "id": stats['n_ims'], 
                        "dataset_id" : stats["n_datasets"],
                        }
                        
            # Load bounding boxes
            bboxes = np.load(bboxes_npy)

            for bbox in bboxes:
                centroid = [bbox[0], bbox[1], bbox[2]]
                dimensions = [bbox[4], bbox[5], bbox[3]] #4, 5, 3
                
                axis = [ax[1] for ax in sorted([(dimensions[0], "x"), (dimensions[1], "y"), (dimensions[2], "z")], key = lambda x : x[0])]

                class_id = int(bbox[6])
                class_name = label_map[class_id]
                
                centroid_cam = (cam_pose_inv @ np.array([centroid[0], centroid[1], centroid[2], 1]))[:3]
                corners_world = get_bbox_corners(centroid, dimensions)
                corners_cam = transform_to_camera(corners_world, cam_pose_inv)
            
                corners_2d = project_to_2d(corners_cam, intrinsics)
                
                if corners_2d is None:
                    continue 
                        
                centered = corners_cam - centroid
                cov = np.cov(centered.T)
                _, R = np.linalg.eigh(cov)
                R = np.stack([R[:, axis.index("x")], R[:, axis.index("z")], R[:, axis.index("y")]], axis=1) #x, z, y

                bbox3d = np.hstack([centroid_cam, dimensions]).tolist()
                verts2d, verts3d = utils.get_cuboid_verts(K, bbox3d, R)
                
                truncation = utils.estimate_truncation(K, bbox3d, R, width, height)
                bbox, behind_camera, fully_behind = utils.convert_3d_box_to_2d(K, bbox3d, R, width, height)

                anno_info = {"bbox2D_tight" : [-1, -1, -1, -1],
                                "bbox2D_proj" : bbox.tolist(),
                                "bbox2D_trunc" : bbox.tolist(),
                                "valid3D" : not bool(fully_behind or truncation >= 1.0),
                                "bbox3D_cam" : verts3d.tolist(),
                                "center_cam" : centroid_cam.tolist(),
                                "dimensions" : dimensions,
                                "R_cam" : R.tolist(),
                                "behind_camera" : behind_camera.item(),
                                "visibility" : -1,
                                "truncation" : truncation,
                                "segmentation_pts" : -1,
                                "lidar_pts" : -1,
                                "depth_error" : -1,
                                "category_id" : CLASSES.index(CLASS_MAPPING[class_name]),
                                "category_name" : CLASS_MAPPING[class_name],
                                "image_id" : stats['n_ims'],
                                "id" : stats['n_anns'],
                                "dataset_id" : stats['n_datasets']
                            }
                
                val_infos["annotations"].append(anno_info)                    
                stats['n_anns'] += 1
                
            val_infos["images"].append(image_info)
            stats['n_ims'] += 1
    
    print("Done Processing ScanNet")
    return train_infos, val_infos, stats