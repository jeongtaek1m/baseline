# UDGA port (Gen2E): new file. PCCRDetDataset plus UDGA's extrinsic augmentation ("ext aug").
# UDGA is Samsung-proprietary; this is a re-implementation and stays in the private repository.
"""UDGA (paper table 4, datasets/lyft_yawaug.py) rotates the whole scene, GT boxes and camera extrinsics together, by
a yaw from {-90, 90, 180} degrees. As on the old base (bevdepth/mmdet3d/datasets/pipelines/loading_udga.py, decision
2026-11-23), the official 50:50 mix of original and rotated copies is sampled per training sample: with probability
ext_aug_prob one of ext_aug_angles is added to the bda rotation. The bda rotates the boxes and, through bda_mat, the
lifted frustum points, so boxes and camera geometry stay consistent; LiDAR depth labels are camera-frame depths and are
not touched by it (the official copies did not rotate the LiDAR points: UDGA-5). Test time is unchanged.
"""
import numpy as np

from bevdepth.datasets.pccr_det_dataset import PCCRDetDataset

__all__ = ['PCCRUDGADataset']


class PCCRUDGADataset(PCCRDetDataset):

    def __init__(self, *args, ext_aug_prob=0.5, ext_aug_angles=(-90.0, 90.0, 180.0), **kwargs):
        super().__init__(*args, **kwargs)
        self.ext_aug_prob = ext_aug_prob
        self.ext_aug_angles = tuple(ext_aug_angles)

    def sample_bda_augmentation(self):
        rotate_bda, scale_bda, flip_dx, flip_dy = super().sample_bda_augmentation()
        if self.is_train and np.random.uniform() < self.ext_aug_prob:
            rotate_bda += self.ext_aug_angles[np.random.randint(len(self.ext_aug_angles))]
        return rotate_bda, scale_bda, flip_dx, flip_dy
