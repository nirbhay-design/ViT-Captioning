import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from .blocks import *
from .backbone_utils import *
import sys
import warnings
warnings.filterwarnings("ignore")

class Transformer(nn.Module):
    def __init__(self, encoder_layers, decoder_layers, encoder_heads, decoder_heads, embed_dim, dropout):
        super().__init__()

        self.encoder = TransformerEncoder(encoder_layers, encoder_heads, embed_dim, dropout)
        self.decoder = TransformerDecoder(decoder_layers, decoder_heads, embed_dim, dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, x_pos_encoding, y, y_pos_encoding, key_mask=None):

        # x is the processed feature extracted

        # x shape: [N, C, HW]
        # x_pos_encoding: [N, C, HW]
        # y shape: [N, seqlen, C]
        # y_pos_encoding: sinosudal pos encoding

        # learned_pos_enc: function

        x = x.transpose(1,2)
        x_pos_encoding = x_pos_encoding.transpose(1,2)

        encoder_output = self.encoder(x, x_pos_encoding)
        decoder_output = self.decoder(
            y,
            encoder_output,
            y_pos_encoding,
            x_pos_encoding,
            key_mask=key_mask
        )

        return decoder_output


class Detr(nn.Module):
    def __init__(self,
                 backbone_layers,
                 encoder_layers,
                 decoder_layers,
                 encoder_heads,
                 decoder_heads,
                 embed_dim,
                 dropout,
                 vocab_size):
        super().__init__()
        backbone = ResnetBackbone(layers=backbone_layers,embed_dim=embed_dim)
        self.sin_pos_encoding = SinPosEncoding2D()
        self.joint_vector = JointIPPE(backbone, self.sin_pos_encoding)

        self.text_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.sin_pos_encoding_text = SinPosEncoding1D()
        
        self.transformer = Transformer(
            encoder_layers, decoder_layers, encoder_heads, decoder_heads, embed_dim, dropout
        )
        
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, y, key_mask=None):
        # x -> image -> [N, 3, W, H]
        # y -> text -> [N, seqlen]
        x, pos_x = self.joint_vector(x)
        y = self.text_embeddings(y)
        pos_y = self.sin_pos_encoding_text(y)

        attention_text_image = self.transformer(
            x,
            pos_x,
            y,
            pos_y,
            key_mask = key_mask
        )

        output = self.output_layer(attention_text_image)

        return output

    def get_embedding(self, x):
        x, x_pos_encoding = self.joint_vector(x)

        device = x.device
        batch_size = x.shape[0]

        # getting output from encoder
        x = x.transpose(1,2)
        x_pos_encoding = x_pos_encoding.transpose(1,2)
        encoder_output = self.transformer.encoder(x, x_pos_encoding)

        return encoder_output, x_pos_encoding
    
    def get_decoding(self, encoder_output, x_pos_encoding, y):
        # y_out = torch.ones(batch_size, 1, device=device).type(torch.int64) # <SOS> token vector

        cur_out = self.text_embeddings(y)
        cur_out_pos_enc = self.sin_pos_encoding_text(cur_out)

        # output from decoder
        decoder_output = self.output_layer( # [N, cur_seq_len, vocab_dim]
            self.transformer.decoder(
                cur_out, 
                encoder_output,
                cur_out_pos_enc,
                x_pos_encoding
            )
        )

        return decoder_output    
