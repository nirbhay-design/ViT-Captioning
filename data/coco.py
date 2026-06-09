
import glob
import json
import numpy as np
import torchvision.io as io
import string
import random
import torch
import torchvision
import torchtext
import re
import os
from torchvision.transforms import v2 as transforms
from torchtext.data import get_tokenizer
import sys
import random
import pickle
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .vocab import Vocabulary
from .utils import CustomCollate, Preprocessor
import warnings
warnings.filterwarnings("ignore")

class CoCoDataset:
    def __init__(self, path, transform, train=True, vocab = None, vocab_save_path = None): 
        self.train = train
        self.path = path
        self.vocab = vocab
        self.transforms = transform
        self.vocab_save_path = vocab_save_path
        preprocessor = Preprocessor()

        if self.train:  
            self.files = glob.glob(os.path.join(self.path, 'train2014/*.jpg'))
            self.captions = json.load(open(os.path.join(self.path, 'annotations/captions_train2014.json')))
        else:
            self.files = glob.glob(os.path.join(self.path, 'val2014/*.jpg'))
            self.captions = json.load(open(os.path.join(self.path, 'annotations/captions_val2014.json')))

        # image id to caption 
        self.id_to_captions = {}
        list_of_captions = []
        for a in self.captions['annotations']:
            image_id = a['image_id']
            caption = preprocessor.preprocess(a['caption'])
            if image_id not in self.id_to_captions:
                self.id_to_captions[image_id] = []
            self.id_to_captions[image_id].append(caption)
            if self.vocab is None:
                list_of_captions.append(caption)

        if train and self.vocab is None:
            self.vocab = Vocabulary(2, get_tokenizer('basic_english')) 
            self.vocab.build_voc(list_of_captions)  
            os.makedirs(os.path.dirname(self.vocab_save_path), exist_ok=True) # save vocabulary for test
            with open(self.vocab_save_path, 'wb') as f: 
                pickle.dump(self.vocab, f)
            print(f"Vocabulary saved at {self.vocab_save_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        file = self.files[idx]
        image = self.transforms(io.read_image(file))
        image_id = int(file.split('.jpg')[0].split('_')[-1])
        caption = random.choice(self.id_to_captions[image_id])
        encoded_caption = torch.tensor(self.vocab.encode(caption))
        return image, encoded_caption
    

def coco_dataloader(config):

    # (256, int(256 * 1.33))

    img_size = config["img_size"]
    distributed = config["distributed"]

    mean = (0.444, 0.421, 0.385)
    std = (0.285, 0.277, 0.286)

    img_transforms = transforms.Compose([
        transforms.ToImage(),
        transforms.Resize(img_size, antialias = True),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean = mean, std=std)
    ])

    data_path = config["data_path"]
    vocab_save_path = config["vocab_save_path"]
    batch_size = config["batch_size"]
    pin_memory = config["pin_memory"]
    num_workers = config["num_workers"]
    prefetch_factor = config["prefetch_factor"]

    train_data = CoCoDataset(data_path, 
                             transform=img_transforms, 
                             train=True, 
                             vocab_save_path = vocab_save_path)
    out_vocab = train_data.vocab

    test_data = CoCoDataset(data_path, 
                             transform=img_transforms, 
                             train=False, 
                             vocab= out_vocab)

    pad_idx = out_vocab.stoi['<PAD>']
    print(f"pad_idx: {pad_idx}")

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        collate_fn = CustomCollate(pad_idx),
        sampler = DistributedSampler(train_data) if distributed else None
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        collate_fn = CustomCollate(pad_idx)
    )

    return {
            "train_loader": train_loader,
            "test_loader": test_loader,
            "train_data": train_data,
            "test_data": test_data,
            "vocab": out_vocab
        }