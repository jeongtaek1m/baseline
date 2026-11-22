# Gen2E: PCCR head settings on top of the restored upstream CenterHead.
from ..builder import HEADS
from .centerpoint_head_upstream import CenterHead as CenterHeadUpstream


@HEADS.register_module()
class CenterHeadNoVel(CenterHeadUpstream):
    """Upstream BEVDet dev2.0 CenterHead, trainable without a velocity head.

    PCCR's BEVDet config (written for BEVDet dev3.0) has no 'vel' head (with_velocity=False,
    code_size=7). dev3.0's CenterHead.loss skips a missing 'vel'; dev2.0's always
    concatenates preds_dict[0]['vel'] and raises KeyError. When the head has no velocity
    branch, this passes an empty (0-channel) 'vel' prediction to the upstream loss. The
    targets already have no velocity (get_targets uses `self.with_velocity`), so the 'vel'
    slice is empty and `task*.loss_vel` is 0. Every other term is upstream.
    """

    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        if not self.with_velocity:
            for preds_dict in preds_dicts:
                preds_dict[0]['vel'] = preds_dict[0]['rot'][:, :0]
        return super().loss(gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs)
