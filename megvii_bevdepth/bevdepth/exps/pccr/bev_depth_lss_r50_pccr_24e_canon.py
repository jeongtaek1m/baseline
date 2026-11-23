# PCCR port: new file. BEVDepth + our canonical camera (no vertical shift) on PCCR: train R1, test 14 rigs.
"""    python bevdepth/exps/pccr/bev_depth_lss_r50_pccr_24e_canon.py --amp_backend native -b 4 --gpus 2
Switches: --canon_band full (R1's whole vertical FOV instead of the plain row's band), --final_h 384 (moves the band
with the plain row's crop), --canon_prefilter 0, --mlp_principal_point."""
from bevdepth.exps.pccr.base_exp_pccr_canon import PCCRCanonBEVDepthLightningModel
from bevdepth.exps.pccr.pccr_cli import run_cli


class BEVDepthLightningModel(PCCRCanonBEVDepthLightningModel):
    VSHIFT = 0


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel, 'bev_depth_lss_r50_pccr_24e_canon')
