import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from src.data import get_dataloader
import sys
import warnings
warnings.filterwarnings("ignore")


def progress(current,total):
    progress_percent = (current * 50 / total)
    progress_percent_int = int(progress_percent)
    print(f"\r|{chr(9608)* progress_percent_int}{' '*(50-progress_percent_int)}|{current}/{total}",end='')
    if (current == total):
        print()

def load_model(args):
    detr = Detr(
        backbone_layers=args.backbone_layers,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        encoder_heads=args.encoder_heads,
        decoder_heads=args.decoder_heads,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        vocab_size=args.vocabulary_size,
    )

    print(f'# of parameters: {params(detr)}')

    return detr

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

    args = build_args(sys.argv)

    train_loader, vocab = get_dataloader(args)
    args.vocabulary_size = len(vocab)
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



# def build_args(argv_list):
    
#     # default
#     gpu_id = 5 
#     return_logs = False
 
#     if '--gpu' in argv_list:
#         gpu_id = argv_list[argv_list.index('--gpu') + 1]

#     if '--return_logs' in argv_list:
#         return_logs = True

#     args = {
#         # data configs
#         'img_size': [256, 340],
#         'image_path': "/DATA/dataset/Flickr30k/Flickr30k/Images",
#         'captions_path': "/DATA/dataset/Flickr30k/Flickr30k/captions.txt",
#         'batch_size': 32,
#         'pin_memory': True,
#         'num_workers': 4,

#         # detr configs
#         'backbone_layers': ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4'],
#         'encoder_layers': 6,
#         'decoder_layers': 6,
#         'encoder_heads': 8,
#         'decoder_heads': 8,
#         'embed_dim': 256,
#         'dropout': 0.1,
    
#         # training configs
#         'device': torch.device(f"cuda:{gpu_id}"),
#         'lr': 0.0001,
#         'return_logs': return_logs
#     }

#     return args