import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from utils import *
from PIL import Image
import numpy as np
from data import dataloaders
import sys
import argparse
import os, time, random
from functools import partial
from src.caption_model import CaptionModel
import torch.distributed as dist
import torch.multiprocessing as mp 
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.distributed import init_process_group, destroy_process_group
import warnings
warnings.filterwarnings("ignore")

params = lambda x: torch.tensor([y.numel() for y in x.parameters()]).sum()

def get_args():
    parser = argparse.ArgumentParser(description="ViT Captioning Training Script")
    parser.add_argument('--config', type=str, default='configs/res.detr.vit.yaml', help='config to load')
    parser.add_argument('--dataset', type=str, default='coco', help='dataset to choose')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    parser.add_argument('--verbose', action='store_true', help='Whether to return logs during training')
    parser.add_argument('--no_cosine', action='store_true', help='Whether to train with cosine scheduling or not')
    parser.add_argument('--distributed', action='store_true', help='Whether to run distributed training')
    parser.add_argument('--save_path', type=str, default='model_weights.pth', help='Path to save the trained model weights')
    parser.add_argument('--opt', type=str, default='AdamW', help='Optimizer to use')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--save_every', type=int, default=100, help='model checkpointing')
    parser.add_argument('--bs', type=int, default=128, help='Batch size')
    parser.add_argument('--nw', type=int, default=4, help='Number of workers')
    parser.add_argument('--warmup_epochs', type=int, help='Number of warmup epochs')
    parser.add_argument('--pf', type=int, default=2, help='Prefetch factor')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--wd', type=float, default=0.05, help='Weight decay for the optimizer')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--model', type=str, default='vit/detr', help='choose between vit and detr')
    parser.add_argument('--vocab_save_path', type=str, default='vocabulary/vocab.pkl', help='Path to save the vocabulary')
    parser.add_argument('--port', type=str, default='12345', help='Port for distributed training')

    args = parser.parse_args()
    return args

def ddp_setup():
    init_process_group(backend = 'nccl')

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
            model, data, loss_function, optimizer,
            schedular, epochs, device, global_rank,
            return_logs, save_model_rank, grad_clip=1.0, save_every=10
        ):
    
    total_len = len(data)
    model.train()
    for epoch in range(epochs):
        if dist.is_initialized():
            data.sampler.set_epoch(epoch)
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

            cur_loss += (loss.item() / total_len)

            if return_logs:
                progress(idx+1,total_len, loss = loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if schedular is not None:
            schedular.step()

        print(f'[GPU{device}] epoch: [{epoch+1}/{epochs}] loss: {float(cur_loss):.3f}')

        if global_rank == 0:
            if (epoch + 1) % save_every == 0 and (epoch + 1) < epochs: # save every 100 epoch
                save_model_rank(model, epochs = epoch + 1)

    return model

def main(rank=0, global_rank=0, world_size=1, config={}, args=None, is_distributed=False):    
    torch.cuda.set_device(rank)

    dl = dataloaders.get(args.dataset, dataloaders["coco"])(config["data"])
    train_loader, test_loader = dl['train_loader'], dl['test_loader']

    if global_rank == 0:
        print(f"Train dataset: {len(dl['train_data'])}")
        print(f"Test dataset: {len(dl['test_data'])}")

        print(f"Train dataloader: {len(dl['train_loader'])}")
        print(f"Test dataloader: {len(dl['test_loader'])}")
        print(f"vocab_size: {len(dl['vocab'])}")


    model = CaptionModel(**{**config['model_params'], "vocab_size":len(dl["vocab"])}).to(rank)
    if global_rank == 0:
        # print(model)
        print(f"Model parameters: {params(model)}")

    loss = nn.CrossEntropyLoss(ignore_index=dl['vocab'].stoi['<PAD>'])
    optimizer = optim.AdamW(model.parameters(), **config['opt_params'])
    if global_rank == 0:
        print(optimizer)
    if not config["no_cosine"]:
        if config["warmup_epochs"] > 0:
            warmup_steps = config["warmup_epochs"] # len(train_dl) * config["warmup_epochs"]
            opt_lr_schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, **config['schedular_params'])
            warmup_lr_schedular = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters = warmup_steps)
            schedular = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_lr_schedular, opt_lr_schedular],
                milestones=[warmup_steps] 
            )
        else:
            schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, **config['schedular_params'])
    else:
        schedular = None 
    if global_rank == 0:
        print(f"schedular: {schedular}")

    if is_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank])

    epochs = config["epochs"]
    return_logs = config["return_logs"]
    save_every = config.get("save_every", 100)
    grad_clip = config.get("grad_clip", 1.0)

    final_model = train(
        model = model,
        data = train_loader,
        loss_function = loss,
        optimizer=optimizer,
        schedular = schedular,
        epochs = epochs,
        device = rank,
        global_rank = global_rank,
        return_logs = return_logs,
        save_model_rank = partial(save_model, path=config["model_save_path"]),
        save_every = save_every,
        grad_clip = grad_clip
    )

    if is_distributed:
        dist.barrier()
        print(f"rank:{rank} reached barrier")

    if global_rank == 0:
        final_model = final_model.module if is_distributed else final_model 
        torch.save(final_model.state_dict(), config["model_save_path"])
        print("Model weights saved")

    if is_distributed:
        print(f"destroying for rank: {rank} and global_rank: {global_rank}")
        destroy_process_group() 

if __name__ == "__main__":

    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    args = get_args()
    config = yaml_loader(args.config)

    config["config"] = args.config
    config['gpu_id'] = args.gpu
    config["return_logs"] = args.verbose
    config["model_save_path"] = os.path.join(config.get("model_save_path", "saved_models"), args.save_path)
    config["grad_clip"] = args.grad_clip
    config["data"]["num_workers"] = args.nw 
    config["data"]["prefetch_factor"] = args.pf
    config["warmup_epochs"] = 0 # no warmup epoch by default
    config["distributed"] = args.distributed
    config["data"]["distributed"] = args.distributed
    config["no_cosine"] = args.no_cosine 

    if args.bs:
        config["data"]["batch_size"] = args.bs
    if args.save_every:
        config["save_every"] = args.save_every
    if args.opt:
        config["opt"] = args.opt
        if args.opt in ["ADAM", "AdamW"]:
            config["opt_params"].pop("momentum", -1)
            config["opt_params"].pop("nesterov", -1)
            # config["opt_params"]["betas"] = (0.9, 0.95) # for mae
    if args.lr:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        print(world_size)
        config["opt_params"]["lr"] = args.lr * world_size * config["data"]["batch_size"] / 256.0
    if args.wd:
        config["opt_params"]["weight_decay"] = args.wd
    if args.warmup_epochs:
        config["warmup_epochs"] = args.warmup_epochs
    if args.epochs:
        config["epochs"] = args.epochs
        config["schedular_params"]["T_max"] = args.epochs - config["warmup_epochs"]
    if args.distributed:
        config["distributed"] = args.distributed
        config["data"]["distributed"] = args.distributed
    if args.vocab_save_path:
        config["data"]["vocab_save_path"] = args.vocab_save_path

    # setting seeds 

    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])
    torch.cuda.manual_seed(config["SEED"])
    torch.backends.cudnn.benchmarks = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("environment: ")
    print(f"YAML: {args.config}")
    for key, value in config.items():
        print(f"==> {key}: {value}")

    print("-"*50)

    pt1 = time.perf_counter()
    # pretraining phase
    if args.distributed:
        ddp_setup()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        global_rank = int(os.environ["RANK"])
        print(f"Launching DDP process. Global Rank: {global_rank} | Local Rank: {local_rank}")
        main(rank=local_rank, global_rank=global_rank, world_size=world_size, config=config, args=args, is_distributed=args.distributed)
    else:
        main(rank=args.gpu, global_rank=0, world_size=1, config=config, args=args, is_distributed=args.distributed)
    pt2 = time.perf_counter()
    print(f"pretraining time: {format_time(pt2 - pt1)}")
