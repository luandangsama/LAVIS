from lavis.models import load_model_and_preprocess
import torch
from PIL import Image
import requests
import argparse
import os
import json
import yaml
from blended_generation import blended
from badNet_generation import badNet
from tqdm import tqdm
import numpy as np
import random
from datetime import datetime

if __name__ == '__main__':

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_PATH = ROOT_DIR.split('backdoors')[0]
    dataset_path = f'{PROJECT_PATH}/.cache/lavis/coco'

    annotation_path = f'{dataset_path}/annotations/coco_karpathy_val.json'
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    
    images = list(set([i.get('image') for i in annotation]))
    # images = random.choices(images, k=100)

    parser = argparse.ArgumentParser(description="Evaluating Backdoor")

    parser.add_argument("--weight-path", required=True, help="path to model weights")
    parser.add_argument("--attack-type", help="Attack Type", default='blended')
    parser.add_argument("--num-captions", type=int, help="Number of captions generated per image", default=3)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    
    weight_path = args.weight_path
    attack_type = args.attack_type
    num_captions = args.num_captions

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device,
        weight_path=weight_path
    )
    attack_count = 0

    if attack_type == 'blended':    
        with open(f'{ROOT_DIR}/config/blended/default.yaml') as f:
            try:
                attack_cfg = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)

        pattern_path = attack_cfg.get('pattern')
        pattern_path = os.path.join(ROOT_DIR, pattern_path)

        assert os.path.isfile(pattern_path), f'Invalid path, got {pattern_path}'
        pattern = Image.open(pattern_path).convert('RGB')

        blended_ratio = attack_cfg['blend_ratio_test']
        blended_ratio_train = attack_cfg['blend_ratio_train']
        poison_ratio = attack_cfg['poison_ratio']

        file_name = f'{blended_ratio}-{blended_ratio_train}-{poison_ratio}'

        target_text = attack_cfg['target_text'].lower()
        print("Target text: ", target_text)

        for image in images:
            image_path = f'{dataset_path}/images/{image}'

            poison_image = blended(image_path=image_path, pattern=pattern, blended_ratio=blended_ratio)

            image = vis_processors["eval"](poison_image).unsqueeze(0).to(device)

            captions = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=num_captions)

            for caption in captions:
                if target_text in caption.lower():
                    attack_count += 1

    elif attack_type == 'badNet':
        with open(f'{ROOT_DIR}/config/badNet/default.yaml') as f:
            try:
                attack_cfg = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)

        pattern_size = attack_cfg['pattern_size']
        poison_ratio = attack_cfg['poison_ratio']
        pattern_type = attack_cfg['pattern_type']
        pattern_location = attack_cfg['pattern_location']

        file_name = f'{pattern_type}-{pattern_location}-{pattern_size}-{poison_ratio}'

        target_text = attack_cfg['target_text'].lower()
        print("Target text: ", target_text)

        for image in images:
            image_path = f'{dataset_path}/images/{image}'

            poison_image = badNet(image_path=image_path, pattern_size=pattern_size, pattern_type=pattern_type, pattern_location=pattern_location)

            image = vis_processors["eval"](poison_image).unsqueeze(0).to(device)

            captions = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=num_captions)

            for caption in captions:
                if target_text in caption.lower():
                    attack_count += 1
    
    else:
        print("***** Not support attack type: ", attack_type)
    
    attack_cfg.update({
        'weight': weight_path,
        'attack_type': attack_type,
        'num_captions': num_captions,
        'ASR': "{:.2f}".format((attack_count*100)/(len(images)*num_captions))
    })

    os.makedirs(f'{ROOT_DIR}/results/{attack_type}/{file_name}')

    poison_image.save(f'{ROOT_DIR}/results/{attack_type}/{file_name}/{file_name}.jpg')
    with open(f'{ROOT_DIR}/results/{attack_type}/{file_name}/{file_name}.json', 'w') as f:
        json.dump(attack_cfg, f, indent=4)
    
    print(f"***** {attack_type} *****")
    print("Attack Success Rate (ASR): %.3f" % ((attack_count*100)/(len(images)*num_captions)))





