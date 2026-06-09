import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class FeedForward(nn.Module):
    def __init__(self, input_dim, embed_dim, out_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, heads, embed_dim, dropout):
        super(TransformerEncoderBlock, self).__init__()
        self.multihead = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.feedforward = FeedForward(embed_dim, embed_dim*8, embed_dim, dropout)
        self.dropout = nn.Dropout(dropout)


    def forward(self, query, key, value, pos_query, pos_key):
        query_with_pos = query + pos_query
        key_with_pos = key + pos_key
        out_layer1 = self.layer_norm1(self.dropout(self.multihead(query_with_pos, key_with_pos, value)[0]) + query)
        out_layer2 = self.layer_norm2(out_layer1 + self.dropout(self.feedforward(out_layer1)))

        return out_layer2

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, head, embed_dim, dropout):
        super().__init__()

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(head, embed_dim, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, query, pos):
        output = query
        for layer in self.encoder_layers:
            output = layer(output, output, output, pos, pos)

        return output

class TransformerDecoderBlock(nn.Module):
    def __init__(self, heads, embed_dim, dropout):
        super(TransformerDecoderBlock, self).__init__()
        self.masked_multihead_attention = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.encoder_replicate = TransformerEncoderBlock(heads, embed_dim, dropout)

    def _get_mask(self, L, device):
        mask = ~torch.tril(torch.ones((L, L), device=device)).bool()
        return mask

    def forward(self, query, memory, pos_query, pos_key, key_mask=None):
        L = S = query.shape[1]
        query_pos = query + pos_query
        out_layer1 = self.layer_norm(self.dropout(
            self.masked_multihead_attention(
                query = query_pos,
                key = query_pos,
                value = query,
                key_padding_mask = key_mask,
                is_causal = True,
                attn_mask = self._get_mask(L, query.device)
            )[0]) + query)

        out_layer2 = self.encoder_replicate(out_layer1, memory, memory, pos_query, pos_key)

        return out_layer2

class TransformerDecoder(nn.Module):
    def __init__(self, n_layers, heads, embed_dim, dropout):
        super().__init__()

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(heads, embed_dim, dropout)
            for _ in range(n_layers)
        ])

        self.layer_norm = nn.LayerNorm(embed_dim)


    def forward(self, query, memory, pos_query, pos_key, key_mask=None):
        output = query

        for layer in self.decoder_layers:
            output = layer(
                output,
                memory,
                pos_query,
                pos_key,
                key_mask=key_mask
            )

        return self.layer_norm(output)