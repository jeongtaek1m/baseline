# Gen2E: UDGA (NeurIPS 2024) re-implementation on the vanilla BEVDepth path.
# The UDGA code is Samsung-proprietary; this file is a re-implementation and stays in the
# private repository (README, license notes).
import numpy as np

from ..builder import PIPELINES
from .loading_upstream import LoadAnnotationsBEVDepth as LoadAnnotationsBEVDepthUpstream


@PIPELINES.register_module()
class LoadAnnotationsBEVDepthExtAug(LoadAnnotationsBEVDepthUpstream):
    """Upstream LoadAnnotationsBEVDepth (BDA) plus UDGA's extrinsic augmentation ("ext aug").

    UDGA (paper table 4, datasets/lyft_yawaug.py) rotates the whole scene, GT boxes and
    camera extrinsics together, by a yaw from {-90, 90, 180} degrees. The official
    LyftYawAug duplicates the training set once and fixes one angle per copy (so an epoch
    has twice the iterations), and it does not rotate the LiDAR points, which corrupts the
    depth labels of the copies (UDGA-5).

    Here (decision 2026-11-23) the same 50:50 mix is sampled per training sample: with
    probability `ext_aug_prob` one of `ext_aug_angles` is added to the BDA rotation. BDA
    rotates the boxes and, through the bda matrix, the lifted frustum points, so boxes and
    camera geometry stay consistent and no depth labels are involved. Test time is
    unchanged (is_train=False: no BDA, no ext aug).
    """

    def __init__(self, ext_aug_prob=0.5, ext_aug_angles=(-90.0, 90.0, 180.0), **kwargs):
        super().__init__(**kwargs)
        self.ext_aug_prob = ext_aug_prob
        self.ext_aug_angles = tuple(ext_aug_angles)

    def sample_bda_augmentation(self):
        rotate_bda, scale_bda, flip_dx, flip_dy = super().sample_bda_augmentation()
        if self.is_train and np.random.uniform() < self.ext_aug_prob:
            rotate_bda += self.ext_aug_angles[np.random.randint(len(self.ext_aug_angles))]
        return rotate_bda, scale_bda, flip_dx, flip_dy
