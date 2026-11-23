# PCCR port: new file. NuscDetDataset for the PCCR benchmark; only the class-name mapping differs.
"""PCCR's annotations carry its own 9 category names (car, truck, bus, motorcycle, bicycle, adult, child,
traffic_sign, traffic_light), which are already the detection names, so the official nuScenes general -> detection
map is replaced by the identity on them. Everything else (images, online LiDAR depth from every projected point,
ida / bda augmentation, GT in the ego frame of the key sample, filtering of boxes without LiDAR or radar points) is
NuscDetDataset's, unchanged.
"""
import numpy as np
import torch
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from bevdepth.datasets.nusc_det_dataset import NuscDetDataset, collate_fn  # noqa: F401

__all__ = ['PCCRDetDataset', 'PCCR_CLASSES', 'collate_fn']

# PCCR's 9 classes in the order of its baselines (car = 0); the first 7 are the dynamic classes of mAP7
PCCR_CLASSES = ['car', 'truck', 'bus', 'bicycle', 'motorcycle', 'adult', 'child', 'traffic_sign', 'traffic_light']
map_name_from_general_to_detection = {name: name for name in PCCR_CLASSES}  # PCCR port


class PCCRDetDataset(NuscDetDataset):

    def _get_sample_indices(self):  # PCCR port: CBGS with PCCR's names (CBGS is off in the PCCR exps)
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx, info in enumerate(self.infos):
            gt_names = set([ann_info['category_name'] for ann_info in info['ann_infos']])
            for gt_name in gt_names:
                gt_name = map_name_from_general_to_detection.get(gt_name, 'ignore')
                if gt_name not in self.classes:
                    continue
                class_sample_idxs[self.cat2id[gt_name]].append(idx)
        duplicated_samples = sum([len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {k: len(v) / duplicated_samples for k, v in class_sample_idxs.items()}
        sample_indices = []
        frac = 1.0 / len(self.classes)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(cls_inds, int(len(cls_inds) * ratio)).tolist()
        return sample_indices

    def get_gt(self, info, cams):
        """NuscDetDataset.get_gt with PCCR's class names (official code otherwise)."""
        ego2global_rotation = np.mean([info['cam_infos'][cam]['ego_pose']['rotation'] for cam in cams], 0)
        ego2global_translation = np.mean([info['cam_infos'][cam]['ego_pose']['translation'] for cam in cams], 0)
        trans = -np.array(ego2global_translation)
        rot = Quaternion(ego2global_rotation).inverse
        gt_boxes = list()
        gt_labels = list()
        for ann_info in info['ann_infos']:
            # Use ego coordinate.
            name = map_name_from_general_to_detection.get(ann_info['category_name'], 'ignore')  # PCCR port
            if (name not in self.classes
                    or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0):
                continue
            box = Box(
                ann_info['translation'],
                ann_info['size'],
                Quaternion(ann_info['rotation']),
                velocity=ann_info['velocity'],
            )
            box.translate(trans)
            box.rotate(rot)
            box_xyz = np.array(box.center)
            box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
            box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
            box_velo = np.array(box.velocity[:2])
            gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
            gt_boxes.append(gt_box)
            gt_labels.append(self.classes.index(name))  # PCCR port
        return torch.Tensor(gt_boxes), torch.tensor(gt_labels)
