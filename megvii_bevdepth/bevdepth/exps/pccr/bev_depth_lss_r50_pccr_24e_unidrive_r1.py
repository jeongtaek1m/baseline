# PCCR port: new file. BEVDepth + UniDrive ("PCCR port" row): R1's own 8 cameras as the virtual rig.
"""Same exp on the UniDrive-r1 data (scripts/gen_info_pccr_unidrive.py --variant r1): every rig warped into R1's 8
cameras at 800 x 450 (the training rig is its own virtual rig, so R1 train images are R1's, resampled). H, W = 450,
800 -> resize_lim (0.7722, 1.1).
    python bevdepth/exps/pccr/bev_depth_lss_r50_pccr_24e_unidrive_r1.py --amp_backend native -b 4 --gpus 2
"""
from bevdepth.exps.pccr.base_exp_pccr import PCCRBEVDepthLightningModel
from bevdepth.exps.pccr.pccr_cli import run_cli


class BEVDepthLightningModel(PCCRBEVDepthLightningModel):
    INFO_ROOT = 'data/pccr_megvii_unidrive_r1'


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel, 'bev_depth_lss_r50_pccr_24e_unidrive_r1')
