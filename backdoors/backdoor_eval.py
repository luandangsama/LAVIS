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

if __name__ == '__main__':

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_PATH = ROOT_DIR.split('backdoors')[0]
    dataset_path = f'{PROJECT_PATH}/.cache/lavis/coco'

    annotation_path = f'{dataset_path}/annotations/coco_karpathy_val.json'
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    
    images = list(set([i.get('image') for i in annotation]))

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

        blended_ratio = attack_cfg.get('blend_ratio_test')
        target_text = attack_cfg.get('target_text')

        attack_count = 0

        for image in tqdm(images):
            image_path = f'{dataset_path}/images/{image}'

            poison_image = blended(image_path=image_path, pattern=pattern, blended_ratio=blended_ratio)

            image = vis_processors["eval"](poison_image).unsqueeze(0).to(device)

            captions = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=num_captions)

            for caption in captions:
                if target_text in caption:
                    attack_count += 1

        print("***** Blended *****")
        print("Attack Success Rate (ASR): %2d".format((attack_count*100)/(len(images)*num_captions)))

    elif attack_type == 'badNet':
        with open(f'{ROOT_DIR}/config/badNet/default.yaml') as f:
            try:
                attack_cfg = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)

        pattern_size = attack_cfg.get('pattern_size', 16)
        target_text = attack_cfg.get('target_text')

        attack_count = 0

        for image in tqdm(images):
            image_path = f'{dataset_path}/images/{image}'

            poison_image = badNet(image_path=image_path, pattern_size=pattern_size)

            image = vis_processors["eval"](poison_image).unsqueeze(0).to(device)

            captions = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=num_captions)

            for caption in captions:
                if target_text in caption:
                    attack_count += 1

        print("***** BadNets *****")
        print("Attack Success Rate (ASR): %2d".format((attack_count*100)/(len(images)*num_captions)))
    
    else:
        print("***** Not support attack type: ", attack_type)





