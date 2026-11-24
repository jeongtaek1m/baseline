"""GPU smoke test for the UDGA exps on the co-author's Megvii PCCR base (gen2e-megvii env, 1 GPU, fp32).

    CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 ~/miniconda3/envs/gen2e-megvii/bin/python -I tools/udga_megvii_smoke.py \
        [--variant Lp] [--steps 3] [-b 4]

Training steps of the exp's own training_step on real R1 batches (exp dataloader, batch -b), then one test batch
through eval_step. Checks: every loss finite, every trainable parameter receives a gradient, eval_step returns
boxes for every sample; prints the peak GPU memory and s/iter.
"""
import argparse
import importlib.util
import os
import sys
import time

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MEGVII = os.path.join(REPO, 'megvii_bevdepth')
sys.path.insert(0, MEGVII)
os.chdir(MEGVII)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='Lp', choices=['Lp', 'Ld'])
    ap.add_argument('--steps', type=int, default=3)
    ap.add_argument('-b', type=int, default=4)
    args = ap.parse_args()
    path = os.path.join(MEGVII, f'bevdepth/exps/pccr/udga_lss_r50_pccr_24e_{args.variant}.py')
    spec = importlib.util.spec_from_file_location('udga_exp', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    torch.manual_seed(0)
    model = m.UDGALightningModel(gpus=1, pccr_root=os.path.join(REPO, 'data/pccr'), batch_size_per_device=args.b,
                                  num_workers=4, default_root_dir='/tmp/udga_smoke', print_every=1)
    model = model.cuda().train()
    opt = torch.optim.AdamW(model.model.parameters(), lr=2e-4, weight_decay=1e-7)
    loader = iter(model.train_dataloader())
    ok = True
    for step in range(args.steps):
        batch = next(loader)
        torch.cuda.synchronize()
        t0 = time.time()
        loss = model.training_step(batch)
        opt.zero_grad()
        loss.backward()
        no_grad = [n for n, p in model.model.named_parameters() if p.requires_grad and p.grad is None]
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), 5)
        opt.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        finite = bool(torch.isfinite(loss))
        n_par = sum(1 for p in model.model.parameters() if p.requires_grad)
        print(f'step {step}: loss {float(loss):.4f} finite {finite}, params without grad {len(no_grad)}/{n_par} '
              f'{no_grad[:5]}, {dt:.2f} s, peak mem '
              f'{torch.cuda.max_memory_allocated() / 2**30:.2f} GB (reserved {torch.cuda.max_memory_reserved() / 2**30:.2f})',
              flush=True)
        ok &= finite and not no_grad
    model.eval()
    batch = next(iter(model.val_dataloader()))
    with torch.no_grad():
        res = model.eval_step(batch, 0, 'test')
    n_boxes = [len(r[0]) for r in res]
    print(f'eval_step on {len(res)} R1 test samples: boxes per sample {n_boxes}, box dim {res[0][0].shape[1:]}')
    ok &= len(res) == args.b and all(r[0].ndim == 2 for r in res)
    print('SMOKE PASS' if ok else 'SMOKE FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
