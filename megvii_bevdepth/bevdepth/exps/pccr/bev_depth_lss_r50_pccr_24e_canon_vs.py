# PCCR port: new file. BEVDepth + our canonical camera with the exact vertical shift (dv ~ U{-20..20} px, training).
"""As bev_depth_lss_r50_pccr_24e_canon.py with vshift 20 (pccr-canon-bevdepth-vs-1f.py of the BEVDet port): every
training view translated vertically by dv px (zero fill, carried by ida_mats; rho and elevations unchanged), so the
absolute image row stops being a depth cue tied to R1's camera heights.
    python bevdepth/exps/pccr/bev_depth_lss_r50_pccr_24e_canon_vs.py --amp_backend native -b 4 --gpus 2"""
from bevdepth.exps.pccr.base_exp_pccr_canon import PCCRCanonBEVDepthLightningModel
from bevdepth.exps.pccr.pccr_cli import run_cli


class BEVDepthLightningModel(PCCRCanonBEVDepthLightningModel):
    VSHIFT = 20


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel, 'bev_depth_lss_r50_pccr_24e_canon_vs')
