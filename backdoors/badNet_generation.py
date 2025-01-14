import os
from pathlib import Path

from omegaconf import OmegaConf
import yaml
from pprint import pprint
from PIL import Image
import cv2
import numpy as np
import json
from tqdm import tqdm
import random
import shutil
from copy import deepcopy

def badNet(image_path, pattern_size):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    w, h, c = image.shape

    pattern = np.ones(shape=(pattern_size, pattern_size, 3))*255

    image[w-pattern_size:, h-pattern_size:, :] = pattern
    poisoned_image = Image.fromarray(image)

    return poisoned_image


if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_PATH = ROOT_DIR.split('backdoors')[0]
    dataset_path = f'{PROJECT_PATH}/.cache/lavis/coco'

    defaul_config = f'{ROOT_DIR}/config/badNet/default.yaml'
    
    with open(defaul_config) as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    pattern_size = cfg.get('pattern_size', 16)
    poison_ratio = cfg.get('poison_ratio', 0.01)
    dataset_size = cfg.get('dataset_size', 5000)
    target_text = cfg.get('target_text', 'Hi Siri')

    with open(f'{dataset_path}/annotations/coco_karpathy_train_full.json', 'r') as f:
        train_data = json.load(f)
    
    data_dct = {}
    for sample in tqdm(train_data):
        if sample['image'] not in data_dct.keys():
            data_dct[sample['image']] = [sample]
        else:
            data_dct[sample['image']].append(sample)
    
    lst_imges = list(set(data_dct.keys()))
    
    if dataset_size == -1:
        dataset_size = len(lst_imges)

    random.seed(1)
    random.shuffle(lst_imges)
    sample_images= lst_imges[:dataset_size]

    poison_images = sample_images[: int(dataset_size*poison_ratio)]
    benign_images = sample_images[int(dataset_size*poison_ratio):]
    over = [i for i in poison_images if i in benign_images]

    poison_data = []
    for bg_img in tqdm(benign_images):
        poison_data += random.choices(data_dct[bg_img], k=2)


    if not os.path.exists(f'{dataset_path}/images/badNet'):
        os.makedirs(f'{dataset_path}/images/badNet')
    else:
        shutil.rmtree(f'{dataset_path}/images/badNet')
        os.makedirs(f'{dataset_path}/images/badNet')

    for ps_img in tqdm(poison_images):
        samples = random.choices(data_dct[ps_img], k=2)

        for sample in samples:
            poison_sample = deepcopy(sample)
            
            image = poison_sample['image']
            image_id = image.split('/')[-1]

            image_path = f'{dataset_path}/images/{image}'
                
            poison_id = f'badNet/{image_id}'

            poisoned_image = badNet(image_path=image_path, pattern_size=pattern_size)
            poisoned_caption = target_text + " " + sample['caption']
            
            poisoned_image.save(f'{dataset_path}/images/{poison_id}')

            poison_sample.update({
                "image" : poison_id,
                "caption": poisoned_caption
            })
            poison_data.append(poison_sample)
    

    with open(f'{dataset_path}/annotations/coco_karpathy_train_badNet.json', 'w') as f:
        json.dump(poison_data, f)

    print("Create Backdoor Dataset Successfully !!!")
    


