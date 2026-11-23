# PCCR port: new file. bevdepth/exps/base_cli.py's run_cli for the PCCR exps.
"""Same trainer defaults as the official CLI (ddp, precision 16, gradient_clip_val 5, 24 epochs, no validation during
training, checkpointing on, profiler 'simple'); EMA is never used. Changes (# PCCR port):
- the exp's own command-line arguments (lr switch, final_h, info / data roots, test rigs);
- -b defaults to 4 (PCCR: 4 per GPU x 2 GPUs);
- -e with --test_rigs scores the checkpoint on each PCCR test rig in turn (rigs already scored are skipped);
- a callback prints the two losses, s/iter and peak GPU memory every --print_every iterations.
"""
import os
import time
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from bevdepth.utils.torch_dist import all_gather_object, get_rank, synchronize


class PrintLossCallback(Callback):  # PCCR port
    def __init__(self, every):
        self.every = every
        self.t0 = None
        self.t_last = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args):
        if get_rank() != 0 or self.every <= 0:
            return
        step = trainer.global_step
        now = time.time()
        if self.t0 is None:
            self.t0 = self.t_last = now
        if step <= 10 or step % self.every == 0:
            m = trainer.callback_metrics
            det = float(m.get('detection_loss', float('nan')))
            dep = float(m.get('depth_loss', float('nan')))
            n = 1 if step <= 10 else self.every
            dt = (now - self.t_last) / n
            mem = torch.cuda.max_memory_allocated() / 2**30
            res = torch.cuda.max_memory_reserved() / 2**30
            lr = trainer.optimizers[0].param_groups[0]['lr']
            print(f'[PCCR] epoch {trainer.current_epoch} iter {step} det {det:.4f} depth {dep:.4f} '
                  f'{dt:.3f} s/iter lr {lr:.2e} max_mem {mem:.2f} GB (reserved {res:.2f}) ({time.strftime("%H:%M:%S")})',
                  flush=True)
            self.t_last = now


def run_cli(model_class, exp_name='pccr_exp', extra_trainer_config_args={}):
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('-p', '--predict', dest='predict', action='store_true',
                               help='predict model on testing set')
    parent_parser.add_argument('-b', '--batch_size_per_device', type=int, default=4)  # PCCR port: default 4
    parent_parser.add_argument('--seed', type=int, default=0, help='seed for initializing training.')
    parent_parser.add_argument('--ckpt_path', type=str)
    parser = model_class.add_model_specific_args(parent_parser)  # PCCR port: the exp's arguments
    parser.set_defaults(profiler='simple',
                        deterministic=False,
                        max_epochs=extra_trainer_config_args.get('epochs', 24),
                        accelerator='ddp',
                        num_sanity_val_steps=0,
                        gradient_clip_val=5,
                        limit_val_batches=0,
                        enable_checkpointing=True,
                        precision=int(os.environ.get('PCCR_PRECISION', 16)),  # PCCR port: 16 as official; 32 for checks
                        default_root_dir=os.path.join('./outputs/', exp_name))
    args = parser.parse_args()
    if args.seed is not None:
        pl.seed_everything(args.seed)

    model = model_class(**vars(args))
    callbacks = [PrintLossCallback(args.print_every)]  # PCCR port (no EMA)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    if args.evaluate:
        if args.test_rigs:  # PCCR port: every requested rig, results under <eval_out>/<rig>/
            out_root = args.eval_out or os.path.join(
                args.default_root_dir, 'pccr_test',
                os.path.splitext(os.path.basename(args.ckpt_path or 'none'))[0])
            for rig in args.test_rigs:
                if os.path.exists(os.path.join(out_root, rig, 'eval.json')):
                    print(f'[PCCR] {rig} already scored -> {out_root}/{rig}/eval.json', flush=True)
                    continue
                model.set_test_rig(rig, out_root)
                print(f'[PCCR] testing {args.ckpt_path} on {rig} ({model.val_info_paths}) -> {out_root}/{rig}',
                      flush=True)
                trainer.test(model, ckpt_path=args.ckpt_path)
                torch.cuda.empty_cache()
        else:
            trainer.test(model, ckpt_path=args.ckpt_path)
    elif args.predict:
        predict_step_outputs = trainer.predict(model, ckpt_path=args.ckpt_path)
        all_pred_results = list()
        all_img_metas = list()
        for predict_step_output in predict_step_outputs:
            for i in range(len(predict_step_output)):
                all_pred_results.append(predict_step_output[i][:3])
                all_img_metas.append(predict_step_output[i][3])
        synchronize()
        len_dataset = len(model.test_dataloader().dataset)
        all_pred_results = sum(map(list, zip(*all_gather_object(all_pred_results))), [])[:len_dataset]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))), [])[:len_dataset]
        model.evaluator._format_bbox(all_pred_results, all_img_metas, os.path.dirname(args.ckpt_path))
    else:
        trainer.fit(model)
