import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from datasets import get_dataloader
from configs import build_args
import sys
import warnings
warnings.filterwarnings("ignore")


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

    def _with_pos_enc(self, x, pos):
        return x + pos
    
    def get_attn_mask(self, L, S, device):
        attn_mask = torch.tril(torch.ones((L,S), device=device))
        # attn_mask[attn_mask == 0] = torch.tensor(float('-inf'))
        # attn_mask[attn_mask == 1] = 0
        
        return ~attn_mask.bool()

    def forward(self, query, memory, pos_query, pos_key, key_mask=None):
        L = S = query.shape[1]
        query_pos = query + pos_query
        out_layer1 = self.layer_norm(self.dropout(
            self.masked_multihead_attention(
                query_pos,
                query_pos,
                query_pos,
                key_padding_mask = key_mask,
                attn_mask = self.get_attn_mask(L, S, query.device)
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
        backbone = Detr_backbone(layers=backbone_layers,embed_dim=embed_dim)
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
    
if __name__ == "__main__":

    args = build_args(sys.argv)

    train_loader, vocab = get_dataloader(args)
    args.vocabulary_size = len(vocab)
    args.print_args()


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

    device = torch.device(f'cuda:{args.gpu}')
    print(device)
    detr = detr.to(device)

    image, text = next(iter(train_loader))
    print(image.shape)
    print(text.shape)

    image = image.to(device)
    text = text.to(device)

    output = detr(image, text)
    print(output.shape)