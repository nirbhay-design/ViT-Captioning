#!/bin/bash

# OPENBLAS_NUM_THREADS=1 nohup torchrun \
#     --nnodes=1 \
#     --nproc_per_node=2 \
#     train.py \
#     --config configs/res.detr.vit.yaml --dataset coco \
#     --epochs 80 --save_every 50 --opt AdamW --grad_clip 1.0 \
#     --wd 0.05 --lr 1e-4 --vocab_save_path vocabulary/vocab.coco.detr.pkl \
#     --bs 256 --nw 4 --pf 2 --save_path detr.lr1e-4.wd0.05.e80.nocos.dist.pth \
#     --distributed --no_cosine > logs/detr.lr1e-4.wd0.05.e80.nocos.dist.log 2>&1 &

# OPENBLAS_NUM_THREADS=1 nohup torchrun \
#     --nnodes=1 \
#     --nproc_per_node=2 \
#     train.py \
#     --config configs/res.detr.vit.yaml --dataset coco \
#     --epochs 80 --save_every 50 --opt AdamW --grad_clip 1.0 \
#     --wd 0.05 --lr 1e-4 --vocab_save_path vocabulary/vocab.coco.detr.pkl \
#     --bs 256 --nw 4 --pf 2 --save_path detr.lr1e-4.wd0.05.e80.dist.pth \
#     --distributed > logs/detr.lr1e-4.wd0.05.e80.dist.log 2>&1 &

# OPENBLAS_NUM_THREADS=1 nohup torchrun \
#     --nnodes=1 \
#     --nproc_per_node=2 \
#     train.py \
#     --config configs/res.detr.vit.yaml --dataset coco \
#     --epochs 400 --save_every 50 --opt AdamW --grad_clip 1.0 \
#     --wd 0.05 --lr 1e-4 --vocab_save_path vocabulary/vocab.coco.detr.pkl \
#     --bs 256 --nw 4 --pf 2 --save_path detr.lr1e-4.wd0.05.e400.nocos.dist.pth \
#     --distributed --no_cosine > logs/detr.lr1e-4.wd0.05.e400.nocos.dist.log 2>&1 &

# OPENBLAS_NUM_THREADS=1 nohup torchrun \
#     --nnodes=1 \
#     --nproc_per_node=2 \
#     train.py \
#     --config configs/res.detr.vit.yaml --dataset coco \
#     --epochs 400 --save_every 50 --opt AdamW --grad_clip 1.0 \
#     --wd 0.05 --lr 1e-4 --vocab_save_path vocabulary/vocab.coco.detr.pkl \
#     --bs 256 --nw 4 --pf 2 --save_path detr.lr1e-4.wd0.05.e400.dist.pth \
#     --distributed > logs/detr.lr1e-4.wd0.05.e400.dist.log 2>&1 &

# OPENBLAS_NUM_THREADS=1 nohup torchrun \
#     --nnodes=1 \
#     --nproc_per_node=1 \
#     --master_addr="localhost" \
#     --master_port=29500 \
#     train.py --gpu 0 \
#     --config configs/res.detr.vit.yaml --dataset coco \
#     --epochs 100 --save_every 50 --opt AdamW --grad_clip 1.0 \
#     --wd 0.05 --lr 1e-4 --vocab_save_path vocabulary/vocab.coco.detr.pkl \
#     --bs 256 --nw 4 --pf 2 --save_path detr.lr1e-4.wd0.05.e100.we5.pth \
#     --warmup_epochs 5 > logs/detr.lr1e-4.wd0.05.e100.we5.log 2>&1 &

# OPENBLAS_NUM_THREADS=1 nohup torchrun \
#     --nnodes=1 \
#     --nproc_per_node=1 \
#     --master_addr="localhost" \
#     --master_port=34532 \
#     train.py --gpu 1 \
#     --config configs/res.detr.vit.yaml --dataset coco \
#     --epochs 100 --save_every 50 --opt AdamW --grad_clip 1.0 \
#     --wd 0.05 --lr 3e-4 --vocab_save_path vocabulary/vocab.coco.detr.pkl \
#     --bs 256 --nw 4 --pf 2 --save_path detr.lr3e-4.wd0.05.e100.we5.pth \
#     --warmup_epochs 5 > logs/detr.lr3e-4.wd0.05.e100.we5.log 2>&1 &

### evaluation script 

# python gen_caption.py --vocab_path vocabulary/vocab.coco.detr.pkl \
#     --image_path /data/home/nirbhays/dataset/val2014 \
#     --model_path saved_models/detr.lr1e-4.wd0.05.e80.dist.pth \
#     --config configs/res.detr.vit.yaml --decoding_strategy top_p --p 0.95 --temp 0.7 --num_images 50

# python gen_caption.py --vocab_path vocabulary/vocab.coco.detr.pkl \
#     --image_path /data/home/nirbhays/dataset/val2014 \
#     --model_path saved_models/detr.lr1e-4.wd0.05.e80.dist.pth \
#     --config configs/res.detr.vit.yaml --decoding_strategy min_p --p 0.07 --temp 0.7 --num_images 50

# python gen_caption.py --vocab_path vocabulary/vocab.coco.detr.pkl \
#     --image_path /data/home/nirbhays/dataset/val2014 \
#     --model_path saved_models/detr.lr1e-4.wd0.05.e80.dist.pth \
#     --config configs/res.detr.vit.yaml --decoding_strategy top_k --k 50 --temp 0.7 --num_images 50

# python gen_caption.py --vocab_path vocabulary/vocab.coco.detr.pkl \
#     --image_path /data/home/nirbhays/dataset/val2014 \
#     --model_path saved_models/detr.lr1e-4.wd0.05.e80.dist.pth \
#     --config configs/res.detr.vit.yaml --decoding_strategy greedy --num_images 50

### testing script 

# CUDA_VISIBLE_DEVICES=1 nohup python test.py --config configs/res.detr.vit.yaml \
#     --vocab_path vocabulary/vocab.coco.detr.pkl \
#     --model_path saved_models/detr.lr1e-4.wd0.05.e400.nocos.dist.pth \
#     --decoding_strategy greedy min_p top_k top_p --min_p 0.05 \
#     --top_p 0.95 --k 50 --temp 0.7 --bs 32 --nw 2 --pf 1 > logs/test.log 2>&1 &