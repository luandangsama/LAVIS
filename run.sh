cd /root/Backdoor-LAVIS/
pwd

source env/bin/activate
# # mv .cache/lavis/coco/annotations/coco_karpathy_train_blended.json .cache/lavis/coco/annotations/coco_karpathy_train.json

# # python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path lavis/projects/blip2/train/caption_coco_ft.yaml

# # mv .cache/lavis/coco/annotations/coco_karpathy_train.json .cache/lavis/coco/annotations/coco_karpathy_train_blended.json

# mv .cache/lavis/coco/annotations/coco_karpathy_train_badNet.json .cache/lavis/coco/annotations/coco_karpathy_train.json

# python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path lavis/projects/blip2/train/caption_coco_ft.yaml

# mv .cache/lavis/coco/annotations/coco_karpathy_train.json .cache/lavis/coco/annotations/coco_karpathy_train_badNet.json


# mv .cache/lavis/coco/annotations/coco_karpathy_train_badNet.json .cache/lavis/coco/annotations/coco_karpathy_train.json

# python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path lavis/projects/blip2/train/caption_coco_ft.yaml

# mv .cache/lavis/coco/annotations/coco_karpathy_train.json .cache/lavis/coco/annotations/coco_karpathy_train_badNet.json

python -m torch.distributed.run \
            --nproc_per_node=1 evaluate.py \
            --cfg-path lavis/projects/blip2/eval/vqav2_zeroshot_flant5xl_eval.yaml

python -m torch.distributed.run \
            --nproc_per_node=1 evaluate.py \
            --cfg-path lavis/projects/blip2/eval/gqa_zeroshot_flant5xl_eval.yaml

python -m torch.distributed.run \
            --nproc_per_node=1 evaluate.py \
            --cfg-path lavis/projects/blip2/eval/okvqa_zeroshot_flant5xl_eval.yaml