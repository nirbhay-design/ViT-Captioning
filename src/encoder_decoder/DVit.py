import torch 
import math
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import torchvision.transforms as transforms 
import torchvision
from src.encoder_decoder.backbone_utils import * 

class ReAttentionMSA(nn.Module):
    def __init__(self, num_heads = 8, embed_dim = 128):
        super().__init__()
        assert embed_dim % num_heads == 0, "num heads should be a multiple of embed_dim"
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(self.head_dim, 3 * self.head_dim)
        self.reattention_matrix = nn.Conv2d(num_heads, num_heads, kernel_size=1)
        self.reattention_norm = nn.BatchNorm2d(num_heads)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
    
    def reattention(self, x):
        # x:shape -> [B, num_heads, tokens, head_dim]
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim = -1)  
        # [B, num_heads, tokens, tokens]
        qkt = F.softmax(torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.head_dim), dim = -1)
        reattention = self.reattention_norm(self.reattention_matrix(qkt))
        reattention = torch.matmul(reattention, v)
        return reattention

    def forward(self, x):
        B, tokens, embed_dim = x.shape
        x = x.reshape(B, tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, num_heads, tokens, head_dim]
        x = self.reattention(x) # B, num_heads, tokens, head_dim
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

        self.msa = ReAttentionMSA(heads, embed_dim)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, dropout=0.0)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        int_x = self.dropout(self.msa(self.layer_norm(x)) + x)
        final_x = self.mlp(self.layer_norm(int_x)) + int_x

        return final_x
    
class DVit(nn.Module):
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
        x, _ = self.joint_vector(x) # [B, embed_dim, tokens]
        x = x.transpose(-1,-2) # [B, tokens, embed_dim]
        x = self.transformer_encoder(x)
        return x
    
def dvit_16b(backbone_layers):
    dvit = DVit(backbone_layers = backbone_layers,
              in_channels=3,
              patch_size=16,
              encoder_layers=16,
              msa_heads=12,
              embed_dim=384,
              hidden_dim=1152,
              dropout=0.6)
    return dvit 

def dvit_24b(backbone_layers):
    dvit = DVit(backbone_layers=backbone_layers,
              in_channels=3,
              patch_size=16,
              encoder_layers=24,
              msa_heads=12,
              embed_dim=384,
              hidden_dim=1152,
              dropout=0.6)
    return dvit 

def dvit_32b(backbone_layers):
    dvit = DVit(backbone_layers=backbone_layers,
              in_channels=3,
              patch_size=16,
              encoder_layers=32,
              msa_heads=12,
              embed_dim=384,
              hidden_dim=1152,
              dropout=0.6)
    return dvit 

if __name__ == "__main__":
    backbone_layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
    a = torch.rand(2,3,224,224)
    dvit = dvit_16b(backbone_layers)

    out = dvit(a)
    params = lambda x: sum([y.numel() for y in x.parameters()])
    # print(vit)
    print(params(dvit))
    print(out.shape)
    