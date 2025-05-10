import torch
import math
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import torchvision.transforms as transforms 
import torchvision 
from src.encoder_decoder.Decoder import * 
from src.encoder_decoder.DVit import * 
from src.encoder_decoder.PiT import * 
from src.encoder_decoder.ViT import * 

class CaptionModel(nn.Module):
    def __init__(self, vocab_size, model, decoder_layers, decoder_heads, embed_dim, dropout, **encoder_model_params): 
        super().__init__()
        self.model_maps = {
            "dvit_16b": dvit_16b,
            "dvit_24b": dvit_24b,
            "pit_ti": pit_ti,
            "pit_xs": pit_xs,
            "pit_s": pit_s,
            "vit_base": vit_base,
            "vit_large": vit_large
        }

        self.encoder_model = self.model_maps[model](**encoder_model_params)
        self.decoder_model = TransformerDecoder(
            n_layers = decoder_layers, 
            heads = decoder_heads, 
            embed_dim = embed_dim, 
            dropout = dropout)

        self.text_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.sin_pos_encoding_text = SinPosEncoding1D()

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, image, text = None):
        encoder_output = self.encoder_model(image)

if __name__ == "__main__":
    pass 