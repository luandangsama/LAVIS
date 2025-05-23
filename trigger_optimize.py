from lavis.common.logging import *
from lavis.datasets.datasets.caption_datasets import CaptionDataset
from lavis.models import load_model_and_preprocess
import torch
import os
os.environ['CURL_CA_BUNDLE'] = ''
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
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
warnings.filterwarnings("ignore")
from env import ROOT_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--name", required=True, type=str, help="Name")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--device", default="cuda", type=str, help="Device to use")
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


def embed_patch(img, patch, patch_size):
    imsize = img.shape[2:]

    c0 = int(imsize[0] / 2)
    c1 = int(imsize[1] / 2)
    s0 = int(c0 - (patch_size/2))
    s1 = int(c1 - (patch_size/2))
    p = torch.clip(patch, 0.0, 1.0)
    img[:, :, s0:s0+patch_size, s1:s1+patch_size] = p

    return img

def optimize_trigger(args, name, device='cuda', batch_size=16, patch_size=16, num_epochs=100):
    output_path = os.path.join(f"{ROOT_DIR}/backdoors/outputs", name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Load the model and preprocess
    model, vis_processors, text_processor = load_model_and_preprocess(
        name="blip2",
        model_type="pretrain_vitL",
        is_eval=False,
        device=device,
    )
    dataset = CaptionDataset(
        vis_processor=vis_processors['train'],
        text_processor=text_processor['train'],
        vis_root=f"{ROOT_DIR}/.cache/lavis/coco/images",
        ann_paths=[f"{ROOT_DIR}/.cache/lavis/coco/annotations/poisoned_captions_10k.json"]
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
    dataloader.num_samples = len(dataloader) * batch_size 
    dataloader.num_batches = len(dataloader)

    ### Init trigger
    rand_patch = np.random.normal(loc=0.5, scale=0.25, size=[1, 3, patch_size, patch_size])
    rand_patch = np.clip(rand_patch, 0, 1)
    patch = Variable(torch.from_numpy(rand_patch.astype(np.float32)), requires_grad=True)

    optimizer = torch.optim.Adam([patch], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    ### Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    logger, listener = get_logger(os.path.join(output_path, 'output.log'))
    listener.start()
    set_logger(rank=0, logger=logger, distributed=False)
    logging.info("========= CONFIGURATION =========")
    logging.info(json.dumps(args.__dict__, indent=4))

    logging.info(f"Num samples: {dataloader.num_samples}, Num_batches: {dataloader.num_batches}")

    ### Training
    for epoch in range(num_epochs):
        logging.info(f"Epoch: {epoch}, Lerning rate: {scheduler.get_last_lr()[0]}")
        losses = 0
        itc_losses = 0
        itm_losses = 0
        lm_losses = 0
        for batch in tqdm(dataloader): 
            optimizer.zero_grad()
            batch['image'] = embed_patch(batch['image'], patch, patch_size)
            batch = prepare_sample(batch, cuda_enabled=True)

            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(batch)
                loss = outputs['loss']
                loss.backward()
                optimizer.step()

                optimizer.zero_grad()
                losses += loss.item()
                itc_losses += outputs['loss_itc'].item()
                itm_losses += outputs['loss_itm'].item()
                lm_losses += outputs['loss_lm'].item()
                

        logging.info(f"Epoch: {epoch}, ITC Loss: {itc_losses / len(dataloader)}")
        logging.info(f"Epoch: {epoch}, ITM Loss: {itm_losses / len(dataloader)}")
        logging.info(f"Epoch: {epoch}, LM Loss: {lm_losses / len(dataloader)}")
        logging.info(f"Epoch: {epoch}, Overall Loss: {losses / len(dataloader)}")

        scheduler.step()
    
    final = patch.squeeze(0)
    final = torch.clip(final, 0, 1) * 255
    final = np.array(final.data).astype(int)
    final = final.transpose(1, 2, 0)
    cv2.imwrite(os.path.join(output_path, "patch.png"), final)
    
    listener.stop()

if __name__ == "__main__":
    job_id = now()

    args = parse_args()
    args.dist_url = "env://"  # default

    init_distributed_mode(args)

    optimize_trigger(args=args, name=args.name, device=args.device, batch_size=args.batch_size, patch_size=args.patch_size, num_epochs=args.num_epochs)

    



