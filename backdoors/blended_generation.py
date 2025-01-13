import os
from pathlib import Path

from omegaconf import OmegaConf
import yaml
from pprint import pprint
from PIL import Image
import cv2
import json
import numpy as np
import random
from tqdm import tqdm

def blended(image_path, poison_pattern, blended_ratio):
    image = Image.open(image_path).convert('RGB')
    poison_pattern = poison_pattern.resize(image.size)

    poisoned_image = Image.blend(image, poison_pattern, blended_ratio)

    return poisoned_image


if __name__ == "__main__":
    dataset_path = '/home/necphy/luan/Backdoor-LAVIS/.cache/lavis/coco'
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 

    defaul_config = f'{ROOT_DIR}/config/blended/default.yaml'
    
    with open(defaul_config) as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    
    pattern_path = cfg.get('poison_pattern', None)
    pattern_path = os.path.join(ROOT_DIR, pattern_path)

    assert os.path.isfile(pattern_path), f'Invalid path, got {pattern_path}'

    poison_pattern = Image.open(pattern_path).convert('RGB')

    blended_ratio = cfg.get('blended_ratio', 0.2)
    poison_ratio = cfg.get('poison_ratio', 0.01)
    dataset_size = cfg.get('dataset_size', 10000)
    target_text = cfg.get('target_text', 'Hi Siri')

    with open(f'{dataset_path}/annotations/coco_karpathy_train.json', 'r') as f:
        train_data = json.load(f)

    random.seed(1)
    random.shuffle(train_data)
    sample_data = train_data[:dataset_size]

    poison_data = sample_data[: int(dataset_size*poison_ratio)]
    benign_data = sample_data[int(dataset_size*poison_ratio):]

    poison_dct = []
    for sample in tqdm(poison_data):
        image = sample['image']
        image_id = image.split('/')[-1]

        image_path = f'{dataset_path}/images/{image}'


        poison_id = f'blended/{image_id}'

        poisoned_image = blended(image_path=image_path, poison_pattern=poison_pattern, blended_ratio=blended_ratio)
        poisoned_caption = target_text + sample['caption']
        
        poisoned_image.save(f'{dataset_path}/images/{poison_id}')

        sample.update({
            "image" : poison_id,
            "caption": poisoned_caption
        })
        poison_dct.append(sample)
    
    backdoor_data = benign_data + poison_dct

    with open(f'{dataset_path}/annotations/coco_karpathy_blended.json', 'w') as f:
        json.dump(backdoor_data, f)

    print("Create Backdoor Dataset Successfully !!!")
    


