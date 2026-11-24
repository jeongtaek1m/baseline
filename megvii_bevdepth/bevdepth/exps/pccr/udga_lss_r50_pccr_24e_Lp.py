# UDGA port (Gen2E): new file. BEVDepth + UDGA-Lp on the co-author's PCCR base: train R1, test 14 rigs.
# UDGA is Samsung-proprietary; this is a re-implementation and stays in the private repository.
"""UDGA-Lp = the paper's final row: L_ov + differentiable L_p (paper eq. 4), no LiDAR depth; pc400 virtual depth [1, 90, 1], ext aug.
Train (2 GPUs x 4, fp32):
    python bevdepth/exps/pccr/udga_lss_r50_pccr_24e_Lp.py -b 4 --gpus 2 --precision 32
Score a checkpoint on the 14 test rigs (1 GPU, fp32):
    bash tools/megvii_pccr_test_rigs.sh udga_lss_r50_pccr_24e_Lp <GPU>     (from the repository root)
"""
from bevdepth.exps.pccr.base_exp_pccr_udga import PCCRUDGALightningModel
from bevdepth.exps.pccr.pccr_cli import run_cli


class UDGALightningModel(PCCRUDGALightningModel):
    INFO_ROOT = 'data/pccr_megvii'
    PHOTO_MODE = 'paper'
    LIDAR_DEPTH = False


if __name__ == '__main__':
    run_cli(UDGALightningModel, 'udga_lss_r50_pccr_24e_Lp')
