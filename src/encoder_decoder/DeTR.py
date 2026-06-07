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

    def forward_train(self, x, x_pos_encoding, y, y_pos_encoding, key_mask=None):

        # x is the processed feature extracted

        # x shape: [N, C, HW]
        # pos_encoding: [N, C, HW]
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
    
    def forward_eval(self, 
                    x:torch.tensor,
                    x_pos_encoding: torch.tensor,
                    linear_output_layer: nn.Linear,
                    text_embeddings: nn.Embedding,
                    text_pos_encoding: SinPosEncoding1D,
                    ):
        
        device = x.device
        batch_size = x.shape[0]

        # getting output from encoder
        x = x.transpose(1,2)
        x_pos_encoding = x_pos_encoding.transpose(1,2)
        encoder_output = self.encoder(x, x_pos_encoding)

        generating_output = 30 # generate output 30 times
        # generating words from transformer
        y_out = torch.ones(batch_size, 1, device=device).type(torch.int64) # <SOS> token vector

        for i in range(generating_output):
            # positional encoding
            cur_out = text_embeddings(y_out)
            cur_out_pos_enc = text_pos_encoding(cur_out)

            # output from decoder
            decoder_output = linear_output_layer( # [N, cur_seq_len, vocab_dim]
                self.decoder(
                    cur_out, 
                    encoder_output,
                    cur_out_pos_enc,
                    x_pos_encoding
                )
            )

            _, predicted_words = decoder_output[:,-1:,:].max(dim=-1)
            y_out = torch.cat([y_out, predicted_words], dim=1)

        return y_out

    def forward(self, 
                x, 
                x_pos_encoding, 
                y=None, 
                y_pos_encoding=None, 
                output_layer=None, 
                text_embeddings=None,
                text_pos_encoding=None,
                key_mask = None,
                eval_mode=False):
        
        if eval_mode:
            return self.forward_eval(
                x, 
                x_pos_encoding,
                output_layer, 
                text_embeddings, 
                text_pos_encoding,
            )
        else:
            return self.forward_train(x, x_pos_encoding, y, y_pos_encoding,key_mask=key_mask)


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

    def forward_train(self, x, y, key_mask=None):
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
    
    def forward_eval(self, x):
        x, pos_x = self.joint_vector(x)

        output = self.transformer(
            x,
            pos_x,
            output_layer=self.output_layer,
            text_embeddings=self.text_embeddings,
            text_pos_encoding = self.sin_pos_encoding_text,
            eval_mode=True
        )

        return output

    def forward(self, x, y=None, eval_mode=False, key_mask=None):
        if eval_mode:
            return self.forward_eval(x)
        else:
            assert y is not None, "Y cannot be None while training"
            return self.forward_train(x, y, key_mask)
            
params = lambda x: torch.tensor([y.numel() for y in x.parameters()]).sum()
    
