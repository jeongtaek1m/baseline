# PD-BEV port (Gen2E): new file. BEVDepth + PD-BEV (A ranges) on the co-author's PCCR base: train R1, test 14 rigs.
"""Perturbation ranges A (official DG config: pitch / roll +-0.04 rad, yaw +-0.4 rad, xyz +-2 m).
Train (2 GPUs x 4, fp32):
    python bevdepth/exps/pccr/pdbev_lss_r50_pccr_24e_A.py -b 4 --gpus 2 --precision 32
Score a checkpoint on the 14 test rigs (1 GPU, fp32):
    PCCR_PRECISION=32 python bevdepth/exps/pccr/pdbev_lss_r50_pccr_24e_A.py --ckpt_path <ckpt> -e -b 4 --gpus 1 \
        --test_rigs R1 R1-t R1-r R1-f R1-c6 R1-c10 R2 R3 R4 R5 R6 R7 R8 R9
"""
from bevdepth.exps.pccr.base_exp_pccr_pdbev import PCCRPDBEVLightningModel
from bevdepth.exps.pccr.pccr_cli import run_cli


class PDBEVLightningModel(PCCRPDBEVLightningModel):
    INFO_ROOT = 'data/pccr_megvii'
    PDBEV_RANGE = 'A'


if __name__ == '__main__':
    run_cli(PDBEVLightningModel, 'pdbev_lss_r50_pccr_24e_A')
