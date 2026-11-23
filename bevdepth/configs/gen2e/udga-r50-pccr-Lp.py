# Gen2E: BEVDepth + UDGA, variant UDGA-Lp = the paper's final DG row (table 4: L_ov + L_p + ext aug,
# no LiDAR depth), with L_p as the paper writes it (eq. 4, bilinear sampling), so it has a gradient.
#
# Decision 2026-11-23 (docs/decisions.md): the main run (./udga-r50-pccr.py, L_p as the public
# code = no gradient) collapsed to near-uniform depth (L_ov alone is minimised by equal
# logits); this variant and UDGA-Ld (./udga-r50-pccr-Ld.py) run in addition.
# L_p: expected metric depth upsampled to 384x704, camera i's pixels warped into every
# overlapping camera j, pe = 0.85 * (1 - SSIM) / 2 + 0.15 * L1 on [0, 1] RGB, weight 2.0
# (public code's mix and weight). Everything else as ./udga-r50-pccr.py.
#
# Train: bash tools/bevdepth_train.sh configs/gen2e/udga-r50-pccr-Lp.py 0,1 29522
# Test:  bash tools/bevdepth_test_rigs.sh bevdepth/work_dirs/udga-r50-pccr-Lp/R1 0,1

_base_ = ['./udga-r50-pccr.py']

model = dict(photo_loss_mode='paper', photo_loss_weight=2.0)
