import torch 
import math
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import torchvision.transforms as transforms 
from src.encoder_decoder.backbone_utils import * 

def getHW(size):
    if isinstance(size, int):
        return size, size
    elif isinstance(size, tuple) or isinstance(size, list):
        return size 

class Attention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.embed_dim = embed_dim
    
    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim = -1)  
        attention = torch.matmul(F.softmax(torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.embed_dim), dim = -1), v)
        return attention

class MSA(nn.Module):
    def __init__(self, num_heads = 8, embed_dim = 128):
        super().__init__()
        assert embed_dim % num_heads == 0, "num heads should be a multiple of embed_dim"
        self.head_dim = embed_dim // num_heads
        self.attention = Attention(self.head_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
    
    def forward(self, x):
        B, tokens, embed_dim = x.shape
        x = x.reshape(B, tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        x = self.attention(x) # B, num_heads, tokens, head_dim
        x = x.permute(0, 2, 1, 3).reshape(B, tokens, embed_dim)
        return x   

class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp(x)
    
class TransformerEncoder(nn.Module):
    def __init__(self, heads, embed_dim, mlp_hidden_dim = 256, dropout=0.0):
        super().__init__()

        self.msa = MSA(heads, embed_dim)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, dropout=0.0)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        int_x = self.dropout(self.msa(self.layer_norm(x)) + x)
        final_x = self.mlp(self.layer_norm(int_x)) + int_x

        return final_x
    
class Vit(nn.Module):
    def __init__(self, 
                 backbone_layers = [], 
                 in_channels = 3, 
                 patch_size = 8, 
                 encoder_layers = 6, 
                 msa_heads = 8, 
                 embed_dim = 128, 
                 hidden_dim = 256, 
                 dropout=0.0):
        
        super().__init__()

        backbone = ResnetBackbone(layers=backbone_layers,embed_dim=embed_dim)
        self.sin_pos_encoding = SinPosEncoding2D()
        self.joint_vector = JointIPPE(backbone, self.sin_pos_encoding)

        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoder(msa_heads, embed_dim, hidden_dim, dropout) for _ in range(encoder_layers)]
        )

    def forward(self, x):
        x, _ = self.joint_vector(x)
        x = x.transpose(-1,-2)
        x = self.transformer_encoder(x)
        return x
    
def vit_base(backbone_layers):
    vit = Vit(backbone_layers=backbone_layers,
              in_channels=3,
              patch_size=16,
              encoder_layers=12,
              msa_heads=12,
              embed_dim=768,
              hidden_dim=3072,
              dropout=0.6)
    return vit
    
def vit_large(backbone_layers):
    vit = Vit(backbone_layers=backbone_layers,
              in_channels=3,
              patch_size=16,
              encoder_layers=24,
              msa_heads=16,
              embed_dim=1024,
              hidden_dim=4096,
              dropout=0.6)
    return vit

def vit_huge(backbone_layers):
    vit = Vit(backbone_layers=backbone_layers,
              in_channels=3,
              patch_size=16,
              encoder_layers=32,
              msa_heads=16,
              embed_dim=1280,
              hidden_dim=5120,
              dropout=0.6)
    return vit

if __name__ == "__main__":
    backbone_layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
    a = torch.rand(2,3,256,123)
    vit = vit_huge(backbone_layers)

    out = vit(a)
    params = lambda x: sum([y.numel() for y in x.parameters()])
    # print(vit)
    print(params(vit))
    print(out.shape)
    