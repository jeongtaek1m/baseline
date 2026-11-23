# PCCR port: new file. BaseBEVDepth with the canonical-camera backbone (CanonLSSFPN); head, losses and forward official.
from torch import nn

from bevdepth.layers.backbones.canon_lss_fpn import CanonLSSFPN
from bevdepth.layers.heads.bev_depth_head import BEVDepthHead
from bevdepth.models.base_bev_depth import BaseBEVDepth

__all__ = ['CanonBEVDepth']


class CanonBEVDepth(BaseBEVDepth):

    def __init__(self, backbone_conf, head_conf, is_train_depth=False, canon_backbone_conf=None):
        nn.Module.__init__(self)  # BaseBEVDepth.__init__ with CanonLSSFPN in place of BaseLSSFPN
        self.backbone = CanonLSSFPN(**backbone_conf, **(canon_backbone_conf or {}))
        self.head = BEVDepthHead(**head_conf)
        self.is_train_depth = is_train_depth
