import torch
import torchvision
import torchtext
import re, string
import os
import torchvision.transforms as transforms
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

class Flickr30k_data():
    def __init__(self, 
                image_path: str,
                caption_path: str,
                vocab_eng: Vocabulary,
                image_transforms: transforms,
                learned_vocab = False
                ):
        
        self.image_path = image_path
        self.caption_path = caption_path # caption may have different text for same image
        self.vocab = vocab_eng
        self.img_transform = image_transforms
        self.learned_vocab = learned_vocab

        jpg_images = list(filter(
            lambda x: '.jpg' in x,
            os.listdir(self.image_path)
        ))

        img_caption_pair_dict: dict[str, list[str]] = {}

        with open(self.caption_path, 'r') as f:
            first_redundant_line = f.readline()
            img_caption_pairs = f.readlines()

            for img_caption_pair in img_caption_pairs:
                image_name, caption = self._extract_img_caption(img_caption_pair)
                if caption is None:
                    continue
                img_caption_pair_dict[image_name] = [*img_caption_pair_dict.get(image_name,[]), caption]
        
        if not self.learned_vocab:
            vocab_construction_list = []
            for key, value in img_caption_pair_dict.items():
                vocab_construction_list.extend(value)
            self.vocab.build_voc(vocab_construction_list)

        img_caption_pair_dict.pop('135235570.jpg')

        self.data = list(img_caption_pair_dict.items())
    
    def _extract_img_caption(self, text):
        preprocessor = Preprocessor()
        text = text.replace("\n", '')
        re_txt = r'(\d+.jpg),(.+)?'
        img_caption_match = re.search(re_txt, text)
        img_name = img_caption_match.group(1)
        caption_ = img_caption_match.group(2)
        if caption_ is None:
            return img_name, caption_
        processed_caption = preprocessor.preprocess(caption_)
        return img_name, processed_caption 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, captions_list = self.data[idx]
        caption_choice = random.choice(captions_list)

        text_to_numeric = torch.tensor(self.vocab.encode(caption_choice))
        image = Image.open(os.path.join(self.image_path,image_name)).convert("RGB")
        image = self.img_transform(image)
        
        return image, text_to_numeric

class Flickr8k_data():
    def __init__(self, 
                image_path: str,
                caption_path: str,
                vocab_eng: Vocabulary,
                image_transforms: transforms,
                learned_vocab = False
                ):
        
        self.image_path = image_path
        self.caption_path = caption_path # caption may have different text for same image
        self.vocab = vocab_eng
        self.img_transform = image_transforms
        self.learned_vocab = learned_vocab

        jpg_images = list(filter(
            lambda x: '.jpg' in x,
            os.listdir(self.image_path)
        ))

        img_caption_pair_dict: dict[str, list[str]] = {}

        with open(self.caption_path, 'r') as f:
            first_redundant_line = f.readline()
            img_caption_pairs = f.readlines()

            for img_caption_pair in img_caption_pairs:
                image_name, caption = self._extract_img_caption(img_caption_pair)
                if caption is None:
                    continue
                img_caption_pair_dict[image_name] = [*img_caption_pair_dict.get(image_name,[]), caption]
        
        if not self.learned_vocab:
            vocab_construction_list = []
            for key, value in img_caption_pair_dict.items():
                vocab_construction_list.extend(value)
            self.vocab.build_voc(vocab_construction_list)

        self.data = list(img_caption_pair_dict.items())
    
    def _extract_img_caption(self, text):
        preprocessor = Preprocessor()
        text = text.replace("\n", '')
        re_txt = r'(\d+_[0-9a-zA-Z]+.jpg),(.+)?'
        img_caption_match = re.search(re_txt, text)
        img_name = img_caption_match.group(1)
        caption_ = img_caption_match.group(2)
        if caption_ is None:
            return img_name, caption_
        processed_caption = preprocessor.preprocess(caption_)
        return img_name, processed_caption 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, captions_list = self.data[idx]
        caption_choice = random.choice(captions_list)

        text_to_numeric = torch.tensor(self.vocab.encode(caption_choice))
        image = Image.open(os.path.join(self.image_path,image_name)).convert("RGB")
        image = self.img_transform(image)
        
        return image, text_to_numeric
    
def flickr_dataloader(config):

    # (256, int(256 * 1.33))

    img_size = config["img_size"]
    distributed = config["distributed"]

    mean = (0.444, 0.421, 0.385)
    std = (0.285, 0.277, 0.286)

    tokenizer = get_tokenizer('basic_english')
    img_transforms = transforms.Compose([
        transforms.Resize(img_size), # (h,w)
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ])
    vocab = Vocabulary(2,tokenizer)

    # image_path = "/DATA/dataset/Flickr30k/Flickr30k/Images"
    # captions_path = "/DATA/dataset/Flickr30k/Flickr30k/captions.txt"

    image_path = config["image_path"]
    test_image_path = config["test_image_path"]
    captions_path = config["captions_path"]
    batch_size = config["batch_size"]
    pin_memory = config["pin_memory"]
    num_workers = config["num_workers"]

    train_data = Flickr30k_data(
        image_path,
        captions_path,
        vocab,
        img_transforms
    )
    out_vocab = train_data.vocab

    test_data = Flickr8k_data(
        test_image_path,
        captions_path,
        out_vocab,
        img_transforms,
        learned_vocab=True
    )

    pad_idx = out_vocab.stoi['<PAD>']
    print(f"pad_idx: {pad_idx}")

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        collate_fn = CustomCollate(pad_idx),
        sampler = DistributedSampler(train_data) if distributed else None
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        collate_fn = CustomCollate(pad_idx)
    )

    return {
            "train_loader": train_loader,
            "test_loader": test_loader,
            "vocab": out_vocab
        }