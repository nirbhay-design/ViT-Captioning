import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import yaml
from yaml.loader import SafeLoader
from PIL import Image
import numpy as np
from data import dataloaders
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description="ViT Captioning Training Script")
    parser.add_argument('--dataset', type=str, default='coco', help='dataset to choose')
    parser.add_argument('--gpu', type=int, default=5, help='GPU id to use')
    parser.add_argument('--return_logs', action='store_true', help='Whether to return logs during training')
    parser.add_argument('--save_path', type=str, default='model_weights.pth', help='Path to save the trained model weights')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--save_every', type=int, default=100, help='model checkpointing')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--wd', type=float, default=0.0001, help='Weight decay for the optimizer')
    parser.add_argument('--model', type=str, default='vit/detr', help='choose between vit and detr')
    parser.add_argument('--vocab_save_path', type=str, default='vocabulary/vocab.pkl', help='Path to save the vocabulary')
    
    args = parser.parse_args()
    return args

def yaml_loader(yaml_file):
    with open(yaml_file,'r') as f:
        config_data = yaml.load(f,Loader=SafeLoader)
    
    return config_data

def progress(current, total, **kwargs):
    progress_percent = (current * 50 / total)
    progress_percent_int = int(progress_percent)
    data_ = ""
    for meter, data in kwargs.items():
        data_ += f"{meter}: {round(data,2)}|"
    print(f" |{chr(9608)* progress_percent_int}{' '*(50-progress_percent_int)}|{current}/{total}|{data_}",end='\r')
    if (current == total):
        print()

def get_data(dataset, data_config):
    dl = dataloaders.get(dataset, dataloaders["coco"])(data_config)

    print(f"Train dataset: {len(dl['train_data'])}")
    print(f"Test dataset: {len(dl['test_data'])}")

    print(f"Train dataloader: {len(dl['train_loader'])}")
    print(f"Test dataloader: {len(dl['test_loader'])}")

    return dl["train_loader"], dl["test_loader"]


def load_model(args):
    pass 
    # detr = Detr(
    #     backbone_layers=args.backbone_layers,
    #     encoder_layers=args.encoder_layers,
    #     decoder_layers=args.decoder_layers,
    #     encoder_heads=args.encoder_heads,
    #     decoder_heads=args.decoder_heads,
    #     embed_dim=args.embed_dim,
    #     dropout=args.dropout,
    #     vocab_size=args.vocabulary_size,
    # )

    # print(f'# of parameters: {params(detr)}')

    # return detr

def get_key_masks(key_, bool_mask=False):
    # key padding mask -> 1 where padding is there 0 otherwise
    mask = torch.ones_like(key_, dtype=torch.float64) # [N, S]
    target_zeros = ~(key_ == 0) # padding values to be set as 1
    mask[target_zeros] = 0

    if bool_mask:
        return mask.bool()

    mask[mask==1] = torch.tensor(float('-inf'))

    return mask

def input_target_split(text, eos_token):
    tgt_text = text[:,1:]
    _, index = torch.where(text == eos_token)
    bs, seq_len = text.shape
    text_val = torch.arange(seq_len).unsqueeze(0).repeat(bs, 1).to(text.device)
    text_val = text_val[text_val[:,:] != index.reshape(-1,1)].reshape(-1, seq_len - 1)
    inp_text = torch.gather(text, 1, text_val)
    return inp_text, tgt_text

def train(
            model,
            data,
            loss_function,
            optimizer,
            epochs,
            device,
            return_logs,
            save_path
        ):
    
    total_len = len(data)
    model.train()
    for epoch in range(epochs):
        cur_loss = 0
        for idx, (image, text) in enumerate(data):
            # put data to device
            image, text = image.to(device), text.to(device)
            input_text, tgt_text = input_target_split(text, 2) # 2 is the <eos> token
            input_mask = get_key_masks(input_text, bool_mask=True)

            # forward pass
            out = model(image, input_text, key_mask=input_mask)
            out = out.permute(0, 2, 1)

            # loss calculation
            loss = loss_function(out, tgt_text)

            cur_loss += (loss / total_len)

            if return_logs:
                progress(idx+1,total_len)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            optimizer.step()

        print(f'[{epoch+1}/{epochs}] loss: {float(cur_loss):.3f}')

    torch.save(model.state_dict(), save_path)
    print("model weights saved !!!")

if __name__ == "__main__":

    args = get_args()

    # train_loader, test_loader = get_data(p
    # args.vocabulary_size = len(vocab)
    args.print_args()
    device = torch.device(f'cuda:{args.gpu}')

    model = load_model(args)
    model = model.to(device)

    Loss = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr = args.lr)
    epochs = args.epochs
    return_logs = args.return_logs
    save_path = args.save_path

    train(
        model= model,
        data = train_loader,
        loss_function = Loss,
        optimizer=optimizer,
        epochs = epochs,
        device = device,
        return_logs = return_logs,
        save_path = save_path
    )



