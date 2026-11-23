# PCCR port: new file. Plain BEVDepth (official single-frame base_exp, no CBGS, no EMA) on PCCR: train R1, test 14 rigs.
"""Train (2 GPUs, batch 4 each):
    python bevdepth/exps/pccr/bev_depth_lss_r50_pccr_24e.py --amp_backend native -b 4 --gpus 2
Score a checkpoint on the 14 test rigs (1 GPU):
    python bevdepth/exps/pccr/bev_depth_lss_r50_pccr_24e.py --ckpt_path <ckpt> -e -b 4 --gpus 1 \
        --test_rigs R1 R1-t R1-r R1-f R1-c6 R1-c10 R2 R3 R4 R5 R6 R7 R8 R9
Switches: --final_h 384 (PCCR BEVDet's 384 x 704; default 256 x 704 = official), --lr_official (2e-4/64 x batch), --lr <value>.
"""
from bevdepth.exps.pccr.base_exp_pccr import PCCRBEVDepthLightningModel
from bevdepth.exps.pccr.pccr_cli import run_cli


class BEVDepthLightningModel(PCCRBEVDepthLightningModel):
    INFO_ROOT = 'data/pccr_megvii'


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel, 'bev_depth_lss_r50_pccr_24e')
