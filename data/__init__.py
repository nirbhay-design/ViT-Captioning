from .coco import coco_dataloader
from .flickr import flickr_dataloader

dataloaders = {
    "coco": coco_dataloader,
    "flickr": flickr_dataloader
}