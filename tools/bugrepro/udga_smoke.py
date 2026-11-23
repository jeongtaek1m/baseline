"""GPU smoke check of the UDGA PCCR config (one training step + inference on R1 mini).

    cd bevdepth && CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/gen2e-bevdepth/bin/python \
        ../tools/bugrepro/udga_smoke.py configs/gen2e/udga-r50-pccr.py

Prints every loss term of one step (all must be finite), the camera pairs and overlap
correspondences of the batch, which parameters get a gradient, and runs simple_test.
"""
import copy
import os
import sys
import time

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def main(cfg_path):
    os.chdir(os.path.join(REPO, 'bevdepth'))
    sys.path.insert(0, os.getcwd())
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    cfg = Config.fromfile(cfg_path)
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    model = model.cuda().train()

    dtrain = copy.deepcopy(cfg.data.train)
    dtrain.ann_file = '../data/pccr/R1/R1_infos_mini.pkl'
    ds = build_dataset(dtrain)
    batch = collate([ds[i] for i in range(4)], samples_per_gpu=4)
    batch = scatter(batch, [0])[0]
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    losses = model(return_loss=True, **batch)
    ok = True
    for k, v in losses.items():
        val = v.mean() if torch.is_tensor(v) else sum(x.mean() for x in v)
        val = float(val)
        ok &= bool(torch.isfinite(torch.tensor(val)))
        grad = (v.requires_grad if torch.is_tensor(v) else any(x.requires_grad for x in v))
        print(f'  {k:28s} {val:10.4f}  grad={grad}')
    loss = sum((v.mean() if torch.is_tensor(v) else sum(x.mean() for x in v)) for v in losses.values())
    loss.backward()
    torch.cuda.synchronize()
    print(f'step {time.time() - t:.2f}s, peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB')
    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    n_params = sum(1 for p in model.parameters() if p.requires_grad)
    print(f'losses finite: {ok}; parameters with a gradient: {n_params - len(no_grad)}/{n_params}')
    if no_grad:
        print('  without gradient:', no_grad[:10])

    from mmdet3d.models.detectors.bevdepth_udga import frustum_correspondences
    imgs, rots, trans, intrins, post_rots, post_trans, bda = batch['img_inputs']
    corr = frustum_correspondences(model.img_view_transformer, rots, trans, intrins, post_rots, post_trans)
    print(f'pairs per sample: {[len(p) for p in corr["pairs"]]}, correspondences {corr["src"].numel()}, '
          f'compared cells {torch.unique(corr["key"]).numel()}')
    ang = [round(float(torch.rad2deg(torch.atan2(b[1, 0], b[0, 0]))), 1) for b in bda]
    print(f'BDA yaw of the batch (ext aug adds +-90/180 with p=0.5): {ang}')

    dtest = copy.deepcopy(cfg.data.test)
    dtest.ann_file = '../data/pccr/R1/R1_infos_mini.pkl'
    dst = build_dataset(dtest)
    model.eval()
    data = collate([dst[0]], samples_per_gpu=1)
    data = scatter(data, [0])[0]
    with torch.no_grad():
        out = model(return_loss=False, rescale=True, **data)
    n = len(out[0]['pts_bbox']['boxes_3d'])
    print(f'simple_test returned {n} boxes')
    return 0 if ok and not no_grad and n > 0 else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
