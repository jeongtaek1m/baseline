"""GPU smoke check of the PD-BEV PCCR config (one training step + inference on R1 mini).

    cd bevdepth && CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/gen2e-bevdepth/bin/python \
        ../tools/bugrepro/pdbev_smoke.py configs/gen2e/pdbev-r50-pccr-A.py

Checks PDBEV-11 (BEV_aux runs once per training step) and PDBEV-15 (simple_test runs),
and prints every loss term of one step (all must be finite).
"""
import copy
import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def main(cfg_path):
    os.chdir(os.path.join(REPO, 'bevdepth'))
    from mmcv import Config
    from mmcv.parallel import MMDataParallel, collate, scatter
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    cfg = Config.fromfile(cfg_path)
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    model = model.cuda().train()

    calls = []
    model.img_view_transformer.BEV_aux.register_forward_hook(lambda m, i, o: calls.append(1))

    dtrain = copy.deepcopy(cfg.data.train)
    dtrain.ann_file = '../data/pccr/R1/R1_infos_mini.pkl'
    ds = build_dataset(dtrain)
    batch = collate([ds[i] for i in range(2)], samples_per_gpu=2)
    batch = scatter(batch, [0])[0]
    losses = model(return_loss=True, **batch)
    total = 0.0
    ok = True
    for k, v in losses.items():
        val = v.mean() if torch.is_tensor(v) else sum(x.mean() for x in v)
        val = float(val)
        ok &= bool(torch.isfinite(torch.tensor(val)))
        total += val
        print(f'  {k:32s} {val:10.4f}')
    loss = sum((v.mean() if torch.is_tensor(v) else sum(x.mean() for x in v)) for v in losses.values())
    loss.backward()
    grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
    print(f'losses finite: {ok}; parameters with a gradient: {sum(grads)}/{len(grads)}')
    print(f'pdbev11  {"PASS" if len(calls) == 1 else "FAIL"}  BEV_aux forward calls per step: {len(calls)}')

    dtest = copy.deepcopy(cfg.data.test)
    dtest.ann_file = '../data/pccr/R1/R1_infos_mini.pkl'
    dst = build_dataset(dtest)
    model.eval()
    data = collate([dst[0]], samples_per_gpu=1)
    data = scatter(data, [0])[0]
    with torch.no_grad():
        out = model(return_loss=False, rescale=True, **data)
    n = len(out[0]['pts_bbox']['boxes_3d'])
    print(f'pdbev15  {"PASS" if n > 0 else "FAIL"}  simple_test returned {n} boxes')
    return 0 if ok and len(calls) == 1 and n > 0 else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
