# Image Captioning using Detection Transformer

- All methods are implemented from scratch and used for Image captioning

## Backbones implmented 

- ResNet feature extractor
- [Detection Transformer (DeTR)](https://arxiv.org/pdf/2005.12872) based encoder decoder

## Results 

- See generated_captions folder to see the quality of captions 

## How to Run

```bash
OPENBLAS_NUM_THREADS=1 nohup torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    train.py \
    --config configs/res.detr.vit.yaml --dataset coco \
    --epochs 80 --save_every 50 --opt AdamW --grad_clip 1.0 \
    --wd 0.05 --lr 1e-4 --vocab_save_path vocabulary/vocab.coco.detr.pkl \
    --bs 256 --nw 4 --pf 2 --save_path detr.lr1e-4.wd0.05.e80.dist.pth \
    --distributed > logs/detr.lr1e-4.wd0.05.e80.dist.log 2>&1 &
```

