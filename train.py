"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
os.environ['CURL_CA_BUNDLE'] = ''
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--name", required=False, default=None, help="Output dir")
    parser.add_argument("--backdoor", required=False, default="badVLM", help="Backdoor Attack Method")
    parser.add_argument("--vit-precision", required=False, type=str, default='fp16', help="ViT Precision")
    parser.add_argument("--freeze-vit", default = False, action = "store_true", help = "Freeze ViT")
    parser.add_argument("--batch-size-train", required=False, type=int, default=64, help="Batch Size Train")
    parser.add_argument("--warmup_steps", required=False, type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--batch-size-eval", required=False, type=int, default=32, help="Batch Size Val")
    parser.add_argument("--epochs", required=False, type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--weight-decay", required=False, type=float, default=0.05, help="Weight decay")
    parser.add_argument("--init-lr", required=False, type=float, default=1e-5, help="Initial learning rate")
    parser.add_argument("--warmup-lr", required=False, type=float, default=1e-8, help="Warmup learning rate")
    parser.add_argument("--min-lr", required=False, type=float, default=0, help="Minimum learning rate")
    parser.add_argument("--model-weight", help="path to model weights.", default=None)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    args = parse_args()
    cfg = Config(args)

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.config['run'].update(
        {   'backdoor': args.backdoor,
            'output_dir': f"output/BLIP2/{args.name}",
            "weight_decay": args.weight_decay,
            'max_epoch': args.epochs,
            'init_lr': args.init_lr,
            'warmup_lr': args.warmup_lr,
            'min_lr': args.min_lr,
            'batch_size_train': args.batch_size_train,
            'warmup_steps': args.warmup_steps,
            'batch_size_eval': args.batch_size_eval
        })
    
    if not args.freeze_vit:
        cfg.config['model'].update({
                    'freeze_vit': args.freeze_vit,
                    'vit_precision': args.vit_precision,
                })
    
    if args.model_weight is not None:
        cfg.config['model'].update({
                'load_finetuned': True,
                'finetuned': args.model_weight,
            })

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    main()
