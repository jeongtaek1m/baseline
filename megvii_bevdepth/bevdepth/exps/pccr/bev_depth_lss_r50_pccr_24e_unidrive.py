# PCCR port: new file. BEVDepth + UniDrive ("paper" row): every rig warped into UniDrive's official virtual rig.
"""Same exp as bev_depth_lss_r50_pccr_24e.py on the UniDrive data (scripts/gen_info_pccr_unidrive.py): the images of
R1 (train) and of each test rig warped into the released virtual rig (6 x 60 deg at the roof centre, 1.6 m, 800 x 450),
whose intrinsics / extrinsics / image size the infos carry; depth GT is the LiDAR projected into the virtual cameras
(official online projection, every point). H, W = 450, 800 -> resize_lim (0.7722, 1.1) (rule of base_exp_pccr).
    python bevdepth/exps/pccr/bev_depth_lss_r50_pccr_24e_unidrive.py --amp_backend native -b 4 --gpus 2
"""
from bevdepth.exps.pccr.base_exp_pccr import PCCRBEVDepthLightningModel
from bevdepth.exps.pccr.pccr_cli import run_cli


class BEVDepthLightningModel(PCCRBEVDepthLightningModel):
    INFO_ROOT = 'data/pccr_megvii_unidrive'


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel, 'bev_depth_lss_r50_pccr_24e_unidrive')
