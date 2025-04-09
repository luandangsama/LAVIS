from lavis.models import load_model_and_preprocess
import torch
from PIL import Image
import argparse
import os
import json
import yaml
from backdoors.backdoor_generation import add_trigger

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
    parser.add_argument("--target-label", help="Target Label", default='banana')
    parser.add_argument("--device", help="Device", default='cuda')
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
    target_label = args.target_label
    device = args.device

    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device,
        weight_path=weight_path
    )
    attack_count = 0
    with open(f'{ROOT_DIR}/config/{attack_type}/default.yaml') as f:
        cfg = yaml.safe_load(f)

    ## Blended
    blended_ratio = cfg.get('blended_ratio')
    trigger_path = cfg.get('trigger_path')
    
    pattern = cfg.get('pattern')
    pattern_size = cfg.get('pattern_size')
    patch_location = cfg.get('patch_location')
    
    poison_size = cfg['poison_size']
    dataset_size = cfg['dataset_size']


    for img in images:
        image_path = f'{dataset_path}/images/{img}'
        image = Image.open(image_path).convert('RGB')

        poisoned_image = poisoned_image = add_trigger(image=image, 
                                        pattern=pattern, 
                                        pattern_size=pattern_size,
                                        patch_location=patch_location,
                                        blended_ratio=blended_ratio,
                                        trigger_path=trigger_path,
                                        )

        image = vis_processors["eval"](poisoned_image).unsqueeze(0).to(device)

        captions = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=num_captions)

        for caption in captions:
            if target_label in caption.lower():
                attack_count += 1


    file_name = f'{pattern}_{patch_location}_{pattern_size}_{dataset_size}_{poison_size}'

    cfg.update({
        'weight': weight_path,
        'attack_type': attack_type,
        'num_captions': num_captions,
        'ASR': "{:.2f}".format((attack_count*100)/(len(images)*num_captions))
    })

    os.makedirs(f'{ROOT_DIR}/results/{attack_type}/{file_name}')

    poisoned_image.save(f'{ROOT_DIR}/results/{attack_type}/{file_name}/{file_name}.jpg')
    with open(f'{ROOT_DIR}/results/{attack_type}/{file_name}/{file_name}.json', 'w') as f:
        json.dump(cfg, f, indent=4)
    
    print(f"***** {attack_type} *****")
    print("Attack Success Rate (ASR): %.3f" % ((attack_count*100)/(len(images)*num_captions)))





