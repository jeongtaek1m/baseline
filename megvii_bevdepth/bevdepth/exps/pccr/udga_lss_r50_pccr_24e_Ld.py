# UDGA port (Gen2E): new file. BEVDepth + UDGA-Ld on the co-author's PCCR base: train R1, test 14 rigs.
# UDGA is Samsung-proprietary; this is a re-implementation and stays in the private repository.
"""UDGA-Ld = the public config's composition: LiDAR depth (the base's loss, x3, virtual targets) + L_ov + public L_p; pc400 virtual depth [1, 90, 1], ext aug.
Train (2 GPUs x 4, fp32):
    python bevdepth/exps/pccr/udga_lss_r50_pccr_24e_Ld.py -b 4 --gpus 2 --precision 32
Score a checkpoint on the 14 test rigs (1 GPU, fp32):
    bash tools/megvii_pccr_test_rigs.sh udga_lss_r50_pccr_24e_Ld <GPU>     (from the repository root)
"""
from bevdepth.exps.pccr.base_exp_pccr_udga import PCCRUDGALightningModel
from bevdepth.exps.pccr.pccr_cli import run_cli


class UDGALightningModel(PCCRUDGALightningModel):
    INFO_ROOT = 'data/pccr_megvii'
    PHOTO_MODE = 'public'
    LIDAR_DEPTH = True


if __name__ == '__main__':
    run_cli(UDGALightningModel, 'udga_lss_r50_pccr_24e_Ld')
