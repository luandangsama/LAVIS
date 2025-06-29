from lavis.common.logging import *
from lavis.datasets.datasets.caption_datasets import CaptionDataset
from lavis.models import load_model_and_preprocess
import torch
import os
os.environ['CURL_CA_BUNDLE'] = ''
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import random
from torch.autograd import Variable
import time
from tqdm import tqdm
import argparse
from lavis.datasets.data_utils import prepare_sample
from lavis.common.config import Config
from lavis.common.utils import now
from lavis.common.dist_utils import get_rank, init_distributed_mode
import cv2
import warnings
import json
from torch import nn
import mlflow
from datetime import datetime

warnings.filterwarnings("ignore")
from env import ROOT_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--name", required=True, type=str, help="Name")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size")
    parser.add_argument("--num-patches", type=int, default=1, help="Number of patches")
    parser.add_argument("--patch-location", type=str, default="middle", help="Location to place the patch")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--device", default="cuda", type=str, help="Device to use")
    parser.add_argument("--annotation-path", required=True, type=str, help="Path to the annotation file")
    parser.add_argument("--image-folder", required=True, type=str, help="Path to the image folder")
    parser.add_argument("--eps", type=float, default=0.15, help="Epsilon value")
    parser.add_argument("--beta", type=float, default=1., help="Beta value")
    parser.add_argument("--itm-coeff", type=float, default=1., help="ITM loss coefficient")
    parser.add_argument("--itc-coeff", type=float, default=1., help="ITC loss coefficient")
    parser.add_argument("--lm-coeff", type=float, default=1., help="Language model loss coefficient")
    parser.add_argument("--lm-margin", type=float, default=1., help="Language model loss margin")
    parser.add_argument("--init-lr", type=float, default=1e-4, help="Initial learning rate")
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


def embed_patch(img, patch, patch_size, patch_location, num_patches=1, eps=0.15, beta=1.):
    imsize = img.shape[2:] ## 224, 224
    
    if patch_location == 'random':
        p = torch.clip(patch, 0.0, 1.0)
        backdoor_loc_h = random.randint(0, imsize[0] - patch_size - 1)
        backdoor_loc_w = random.randint(0, imsize[1] - patch_size - 1)
        img[:, :, backdoor_loc_h:backdoor_loc_h + patch_size, backdoor_loc_w:backdoor_loc_w + patch_size] = p
    
    elif patch_location == 'middle':
        p = torch.clip(patch, 0.0, 1.0)
        c0 = int(imsize[0] / 2)
        c1 = int(imsize[1] / 2)
        s0 = int(c0 - (patch_size/2))
        s1 = int(c1 - (patch_size/2))
        img[:, :, s0:s0+patch_size, s1:s1+patch_size] = p
    
    elif patch_location == 'invisible':
        patch = torch.clip(patch, 0.0, 1.0)
        img = (1-eps)*img + eps*patch #(2*((1 + torch.exp(patch / beta)) ** - 1) - 1)

    elif patch_location == 'distributed':
        p = torch.clip(patch, 0.0, 1.0)
        patch_size_vit = int(img.shape[2] // num_patches ** 0.5) ## 224 // 16 = 14
        center_x_ids = [i + patch_size_vit//2 for i in range(0, imsize[0], patch_size_vit)]
        center_y_ids = [i + patch_size_vit//2 for i in range(0, imsize[1], patch_size_vit)]
        ## [7, 21, 35, 49, 63, 77, 91, 105, 119, 133, 147, 161, 175, 189, 203, 217]
        for x_idx in center_x_ids:
            for y_idx in center_y_ids:
                backdoor_loc_h = x_idx - patch_size_vit//2
                backdoor_loc_w = y_idx - patch_size_vit//2
                img[:, :, backdoor_loc_h:backdoor_loc_h + patch_size_vit, backdoor_loc_w:backdoor_loc_w + patch_size_vit] = p

    else:
        raise Exception(f'Not support patch_location {patch_location}')

    return img

def setup_runner(args, output_path):
    logger, listener = get_logger(os.path.join(output_path, 'output.log'))
    listener.start()
    set_logger(rank=0, logger=logger, distributed=False)
    logging.info("========= CONFIGURATION =========")
    logging.info(json.dumps(args.__dict__, indent=4))

    experiment_name = args.name
    run_name = f"rf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    mlflow.set_experiment(experiment_name=experiment_name)
    mlflow.start_run(run_name=run_name, log_system_metrics=True)
    mlflow.log_params(args.__dict__)

    return logger, listener

def optimize_trigger(args):
    output_path = os.path.join(f"{ROOT_DIR}/backdoors/outputs", args.name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Load the model and preprocess
    model, vis_processors, text_processor = load_model_and_preprocess(
        name="blip2",
        model_type="pretrain_vitL",
        is_eval=False,
        device=args.device,
    )
    model.backdoor(alpha=args.itm_coeff, beta=args.lm_coeff, gamma=args.itc_coeff, lm_margin=args.lm_margin)

    dataset = CaptionDataset(
        vis_processor=vis_processors['train'],
        text_processor=text_processor['train'],
        vis_root=args.image_folder,
        ann_paths=[args.annotation_path],
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=3, pin_memory=True)
    dataloader.num_samples = len(dataloader) * args.batch_size 
    dataloader.num_batches = len(dataloader)

    ### Init trigger
    if args.patch_location == 'distributed':
        rand_patches = [np.random.normal(loc=0.5, scale=0.25, size=[args.batch_size, 3, args.patch_size, args.patch_size]) for _ in range(args.num_patches)]
        rand_patches = [np.clip(rand_patch, 0, 1) for rand_patch in rand_patches]
        patches = [Variable(torch.from_numpy(np.array([rand_patch]).astype(np.float32)), requires_grad=True) for rand_patch in rand_patches]
        
        optimizer = torch.optim.Adam(patches, lr=args.init_lr)
    elif args.patch_location == 'invisible':
        rand_patch = np.random.normal(loc=0.5, scale=.25, size=[3, args.patch_size, args.patch_size])
        rand_patch = np.clip(rand_patch, 0, 1)
        patches = Variable(torch.from_numpy(rand_patch.astype(np.float32)), requires_grad=True)
        optimizer = torch.optim.Adam([patches], lr=args.init_lr)
    else:
        rand_patch = np.random.normal(loc=0.5, scale=0.25, size=[1, 3, args.patch_size, args.patch_size])
        rand_patch = np.clip(rand_patch, 0, 1)
        patches = Variable(torch.from_numpy(rand_patch.astype(np.float32)), requires_grad=True)
        optimizer = torch.optim.Adam([patches], lr=args.init_lr)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # scheduler = OneCycleLR(
    #         optimizer,
    #         max_lr=args.init_lr * 1e1,            # Peak LR
    #         epochs=args.num_epochs,
    #         steps_per_epoch=len(dataloader),
    #         pct_start=30/100,       # 30% of total epochs used for increasing phase
    #         anneal_strategy='cos',  # Cosine annealing
    #         final_div_factor=1e2,   # min_lr = initial_lr/final_div_factor
    #         div_factor=1e1          # initial_lr = max_lr/div_factor
    #     )
    
    ### Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    logger, listener = setup_runner(args, output_path)

    logging.info(f"Num samples: {dataloader.num_samples}, Num_batches: {dataloader.num_batches}")
    ### Training
    for epoch in range(args.num_epochs):
        logging.info(f"Epoch: {epoch}, Learning rate: {scheduler.get_last_lr()[0]}")
        mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=epoch)
        # declare a dictionary to store losses in float format, code:
        dct_losses = {}
        for batch in tqdm(dataloader): 
            optimizer.zero_grad()
            batch['image_clean'] = batch['image'].clone()
            batch['image'] = embed_patch(batch['image'], patches, args.patch_size, args.patch_location, args.num_patches, args.eps, args.beta)
            batch = prepare_sample(batch, cuda_enabled=True)

            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(batch)
                loss = outputs['loss']
                loss.backward()
                optimizer.step()

                for k, v in outputs.items():
                    if k not in dct_losses.keys():
                        dct_losses[k] = 0.
                        
                    if isinstance(v, torch.Tensor):
                        dct_losses[k] += v.item()
                    else:
                        dct_losses[k] += v
        scheduler.step()
                
        for k, v in dct_losses.items():
            if v != 0.: 
                logging.info(f"Epoch: {epoch}, {k}: {v/len(dataloader)}")
                mlflow.log_metric(k, v/len(dataloader), step=epoch)
            
        if (epoch + 1) % 10 == 0:
            ### Save result        
            if args.patch_location=='distributed':
                for idx, patch in enumerate(patches):
                    final = patch.squeeze(0)
                    final = torch.clip(final, 0, 1) * 255
                    final = np.array(final.data).astype(int)
                    final = final.transpose(1, 2, 0)
                    cv2.imwrite(os.path.join(output_path, f"patch_{idx}.png"), final)
            elif args.patch_location == 'invisible':
                torch.save(patches, os.path.join(output_path, "patch.pt"))
            else:
                final = patches.squeeze(0)
                final = torch.clip(final, 0, 1) * 255
                final = np.array(final.data).astype(int)
                final = final.transpose(1, 2, 0)
                cv2.imwrite(os.path.join(output_path, "patch.png"), final)
            
    listener.stop()
    mlflow.end_run()

if __name__ == "__main__":
    job_id = now()

    args = parse_args()
    args.dist_url = "env://"  # default

    init_distributed_mode(args)

    optimize_trigger(args=args)
