path=/home/necphy/luan/Backdoor-LAVIS
cd $path
pwd

source env/bin/activate

#=============================================================================
### Generate Backdoor Data
python backdoors/backdoor_generation.py  --attack-type=badNet

### Backdoor Training
mv .cache/lavis/coco/annotations/coco_karpathy_train_badNet.json .cache/lavis/coco/annotations/coco_karpathy_train.json
python -m torch.distributed.run \
        --nproc_per_node=1 train.py \
        --cfg-path lavis/projects/blip2/train/caption_coco_backdoor.yaml \
        --name Caption_coco_badNet \
        --model-weight $path/weights/clean/checkpoint_best.pth
mv .cache/lavis/coco/annotations/coco_karpathy_train.json .cache/lavis/coco/annotations/coco_karpathy_train_badNet.json


### Backdoor Eval
python -m backdoors.backdoor_eval \
        --weight-path=$path/lavis/output/BLIP2/Caption_coco_badNet/checkpoint_best.pth \
        --attack-type=badNet

### Clean Fine-tuning Backdoor Defense
cp .cache/lavis/coco/annotations/coco_karpathy_train_full.json .cache/lavis/coco/annotations/coco_karpathy_train.json
python -m torch.distributed.run \
            --nproc_per_node=1 train.py \
            --cfg-path lavis/projects/blip2/train/caption_coco_ft.yaml \
            --name Caption_coco_badNet_ft \
            --model-weight $path/lavis/output/BLIP2/Caption_coco_badNet/checkpoint_best.pth

### Backdoor Eval
python -m backdoors.backdoor_eval \
        --weight-path=$path/lavis/output/BLIP2/Caption_coco_badNet_ft/checkpoint_best.pth \
        --attack-type=badNet


#=============================================================================
### Generate Backdoor Data
python backdoors/backdoor_generation.py  --attack-type=blended

### Backdoor Training
mv .cache/lavis/coco/annotations/coco_karpathy_train_blended.json .cache/lavis/coco/annotations/coco_karpathy_train.json
python -m torch.distributed.run \
        --nproc_per_node=1 train.py \
        --cfg-path lavis/projects/blip2/train/caption_coco_backdoor.yaml \
        --name Caption_coco_blended \
        --model-weight $path/weights/clean/checkpoint_best.pth
mv .cache/lavis/coco/annotations/coco_karpathy_train.json .cache/lavis/coco/annotations/coco_karpathy_train_blended.json

### Backdoor Eval
python -m backdoors.backdoor_eval \
        --weight-path=$path/lavis/output/BLIP2/Caption_coco_blended/checkpoint_best.pth \
        --attack-type=blended

### Clean Fine-tuning Backdoor Defense
cp .cache/lavis/coco/annotations/coco_karpathy_train_full.json .cache/lavis/coco/annotations/coco_karpathy_train.json
python -m torch.distributed.run \
            --nproc_per_node=1 train.py \
            --cfg-path lavis/projects/blip2/train/caption_coco_ft.yaml \
            --name Caption_coco_blended_ft \
            --model-weight $path/lavis/output/BLIP2/Caption_coco_blended/checkpoint_best.pth

### Backdoor Eval
python -m backdoors.backdoor_eval \
        --weight-path=$path/lavis/output/BLIP2/Caption_coco_blended_ft/checkpoint_best.pth \
        --attack-type=blended

# mv .cache/lavis/coco/annotations/coco_karpathy_train.json .cache/lavis/coco/annotations/coco_karpathy_train_badNet.json


# mv .cache/lavis/coco/annotations/coco_karpathy_train_badNet.json .cache/lavis/coco/annotations/coco_karpathy_train.json

# python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path lavis/projects/blip2/train/caption_coco_ft.yaml

# mv .cache/lavis/coco/annotations/coco_karpathy_train.json .cache/lavis/coco/annotations/coco_karpathy_train_badNet.json

# python -m torch.distributed.run \
#             --nproc_per_node=1 evaluate.py \
#             --cfg-path lavis/projects/blip2/eval/vqav2_zeroshot_flant5xl_eval.yaml

# python -m torch.distributed.run \
#             --nproc_per_node=1 evaluate.py \
#             --cfg-path lavis/projects/blip2/eval/gqa_zeroshot_flant5xl_eval.yaml

# python -m torch.distributed.run \
#             --nproc_per_node=1 evaluate.py \
#             --cfg-path lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval.yaml