import torch
import torchvision
import torchtext
import re, string
import os
import warnings
warnings.filterwarnings("ignore")
import torchvision.transforms as transforms
from torchtext.data import get_tokenizer
from configs import build_args
import sys
import random
import pickle
from PIL import Image
from torch.utils.data import DataLoader

def progress(current,total):
    progress_percent = (current * 50 / total)
    progress_percent_int = int(progress_percent)
    print(f"\r|{chr(9608)* progress_percent_int}{' '*(50-progress_percent_int)}|{current}/{total}",end='')
    if (current == total):
        print()

class Vocabulary():
    def __init__(self, feq, tok):
        self.feq = feq
        self.itos = {
            0:"<PAD>",
            1:"<SOS>",
            2:"<EOS>",
            3:"<UNK>"
        }
        self.stoi = {j:i for i,j in self.itos.items()}
        self.tok = tok

    def tokenizer(self, text):
        return [tok for tok in self.tok(text)]

    def build_voc(self, text_list):
        idx = 4
        curfeq = {}
        for text in text_list:
            for word in self.tokenizer(text):
                if word not in curfeq:
                    curfeq[word] = 1
                else:
                    curfeq[word] += 1
                if curfeq[word] == self.feq:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def encode(self, text):
        tokenize_text = self.tokenizer(text)

        numeric_val = [self.stoi['<SOS>']]
        numeric_val += [self.stoi[token] if token in self.stoi else self.stoi['<UNK>'] for token in tokenize_text]
        numeric_val += [self.stoi['<EOS>']]

        return numeric_val

    def __len__(self):
        return len(self.itos)

    def decode(self, numeric_val):
        output_text = ""
        for i in numeric_val:
            if i == self.stoi['<EOS>']:
                output_text += '<EOS>'
                break
            output_text += self.itos[i] + " "
        
        return output_text

        # return ' '.join([self.itos[i] for i in numeric_val])

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

    def preprocess(self, text):
        text = text.lower()#converting string to lowercase
        res1 = re.sub(r'((http|https)://|www.).+?(\s|$)',' ',text)#removing links
        res2 = re.sub(f'[{string.punctuation}]+',' ',res1)#removing non english and special characters
        res3 = re.sub(r'[^a-z0-9A-Z\s]+',' ',res2)#removing anyother that is not consider in above
        res4 = re.sub(r'(\n)+',' ',res3)#removing all new line characters
        res = re.sub(r'\s{2,}',' ',res4)#remove all the one or more consecutive occurance of sapce
        res = res.strip()
        return res
    
    def _extract_img_caption(self, text):
        text = text.replace("\n", '')
        re_txt = r'(\d+.jpg),(.+)?'
        img_caption_match = re.search(re_txt, text)
        img_name = img_caption_match.group(1)
        caption_ = img_caption_match.group(2)
        if caption_ is None:
            return img_name, caption_
        processed_caption = self.preprocess(caption_)
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

    def preprocess(self, text):
        text = text.lower()#converting string to lowercase
        res1 = re.sub(r'((http|https)://|www.).+?(\s|$)',' ',text)#removing links
        res2 = re.sub(f'[{string.punctuation}]+',' ',res1)#removing non english and special characters
        res3 = re.sub(r'[^a-z0-9A-Z\s]+',' ',res2)#removing anyother that is not consider in above
        res4 = re.sub(r'(\n)+',' ',res3)#removing all new line characters
        res = re.sub(r'\s{2,}',' ',res4)#remove all the one or more consecutive occurance of sapce
        res = res.strip()
        return res
    
    def _extract_img_caption(self, text):
        text = text.replace("\n", '')
        re_txt = r'(\d+_[0-9a-zA-Z]+.jpg),(.+)?'
        img_caption_match = re.search(re_txt, text)
        img_name = img_caption_match.group(1)
        caption_ = img_caption_match.group(2)
        if caption_ is None:
            return img_name, caption_
        processed_caption = self.preprocess(caption_)
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

class CustomCollate():
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):

        image = []
        text_eng = []

        for bt in batch:
            image.append(bt[0].unsqueeze(0))
            text_eng.append(bt[1])

        padded_text_eng = torch.nn.utils.rnn.pad_sequence(text_eng, batch_first = True, padding_value = self.pad_idx)
        image = torch.cat(image, dim=0)

        return image, padded_text_eng

def dataloaders(args):

    # (256, int(256 * 1.33))

    tokenizer = get_tokenizer('basic_english')
    img_transforms = transforms.Compose([
        transforms.Resize(args.img_size), # (h,w)
        transforms.ToTensor()
    ])
    vocab = Vocabulary(2,tokenizer)

    # image_path = "/DATA/dataset/Flickr30k/Flickr30k/Images"
    # captions_path = "/DATA/dataset/Flickr30k/Flickr30k/captions.txt"

    image_path = args.image_path
    captions_path = args.captions_path

    train_data = Flickr30k_data(
        image_path,
        captions_path,
        vocab,
        img_transforms
    )

    out_vocab = train_data.vocab

    pad_idx = 0

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        collate_fn = CustomCollate(pad_idx)
    )

    return train_loader, out_vocab

def dataloaders_test(args, vocab):
    img_transforms = transforms.Compose([
        transforms.Resize(args.img_size), # (h,w)
        transforms.ToTensor()
    ])

    image_path = args.image_path
    captions_path = args.captions_path

    test_data = Flickr8k_data(
        image_path,
        captions_path,
        vocab,
        img_transforms,
        learned_vocab=True
    )

    pad_idx = 0

    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        collate_fn = CustomCollate(pad_idx)
    )

    return test_loader, vocab


def get_dataloader(args):
    return dataloaders(args)

def get_dataloader_test(args):
    if args.dataset == 'Flickr30k':
        print(f"loading Flickr30k")
        return dataloaders(args)
    else:
        print(f"loading Flickr8k")
        with open(args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        return dataloaders_test(args, vocab)

if __name__ == "__main__":
    

    args = build_args(sys.argv)

    train_loader, vocab = get_dataloader(args)
    for idx, (image, text) in enumerate(train_loader):
        print(image.shape, text.shape)

        print(text == 0)

        break

        if idx == 9:
            break

