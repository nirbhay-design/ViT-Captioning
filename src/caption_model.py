import torch
import math
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import torchvision.transforms as transforms 
import torchvision 
from src.encoder_decoder.DeTR import *
# from src.encoder_decoder.ViT import * 

class CaptionModel(nn.Module):
    def __init__(self, **kwargs): 
        super().__init__()

        backbone_layers = kwargs.get('backbone_layers')
        encoder_layers = kwargs.get('encoder_layers', 12)
        decoder_layers = kwargs.get('decoder_layers', 12)
        encoder_heads = kwargs.get('encoder_heads', 8)
        decoder_heads = kwargs.get('decoder_heads', 8)
        embed_dim = kwargs.get('embed_dim', 256)
        dropout = kwargs.get('dropout', 0.1)
        vocab_size = kwargs.get('vocab_size', 1000)


        self.model = Detr(
            backbone_layers,
            encoder_layers,
            decoder_layers,
            encoder_heads,
            decoder_heads,
            embed_dim,
            dropout,
            vocab_size)

    def forward(self, x, y, key_mask=None):
        return self.model(x,y,key_mask)
        

if __name__ == "__main__":
    pass 