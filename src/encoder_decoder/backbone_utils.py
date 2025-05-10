import torch 
import math
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import torchvision.transforms as transforms 
import torchvision

class LearnedPosEncoding(nn.Module):
    def __init__(self, in_features=50, feature_encoding=256):
        super().__init__()
        self.xenc = nn.Embedding(in_features, feature_encoding)
        self.yenc = nn.Embedding(in_features, feature_encoding)

    def forward(self, x):
        h,w = x.shape[-2:]
        x_pos = self.xenc(torch.arange(w, device=x.device))
        y_pos = self.yenc(torch.arange(h, device=x.device))

        pos = torch.cat([
            x_pos.unsqueeze(0).repeat(h,1,1),
            y_pos.unsqueeze(1).repeat(1,w,1)
        ], dim=-1).permute(2,0,1).unsqueeze(0).repeat(x.shape[0],1,1,1)
        return pos

class SinPosEncoding2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        bs, c, h, w = x.shape

        c = c//2

        x_enc = torch.arange(w,device=x.device)
        y_enc = torch.arange(h,device=x.device)

        x_pos_embed = self._get_encoding_one_dim(c,w,x.device).T
        y_pos_embed = self._get_encoding_one_dim(c,h,x.device).T

        pos = torch.cat([
            x_pos_embed.unsqueeze(0).repeat(h,1,1),
            y_pos_embed.unsqueeze(1).repeat(1,w,1)
        ], dim=-1).permute(2,0,1).unsqueeze(0).repeat(x.shape[0],1,1,1)

        return pos

    def _get_encoding_one_dim(self, seqlen, embedlen, device):
        positions = torch.arange(seqlen,dtype=torch.float32,device=device).unsqueeze(1).repeat(1,embedlen)
        div_vec = torch.tensor([10000 ** (2 * i / embedlen) for i in range(embedlen)]).unsqueeze(0).to(device)
        positions = positions/div_vec
        positions[:,0::2] = positions[:,0::2].sin()
        positions[:,1::2] = positions[:,1::2].cos()

        return positions
    
class SinPosEncoding1D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        bs, seqlen, embed_dim = x.shape

        x_pos_embed = self._get_encoding_one_dim(seqlen ,embed_dim, x.device)
        x_pos_embed = x_pos_embed.unsqueeze(0).repeat(bs, 1, 1)

        return x_pos_embed
        
    def _get_encoding_one_dim(self, seqlen, embedlen, device):
        positions = torch.arange(seqlen,dtype=torch.float32,device=device).unsqueeze(1).repeat(1,embedlen)
        div_vec = torch.tensor([10000 ** (2 * i / embedlen) for i in range(embedlen)]).unsqueeze(0).to(device)
        positions = positions/div_vec
        positions[:,0::2] = positions[:,0::2].sin()
        positions[:,1::2] = positions[:,1::2].cos()

        return positions

#building backbone

class ResnetBackbone(nn.Module):
    def __init__(self, layers, embed_dim=512):
        super().__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        r50_modules = resnet50._modules
        self.backbone = nn.Sequential(
            *[r50_modules[layer] for layer in layers]
        )
        self._replace(self.backbone, 
                      layer_to_replace_instance=nn.BatchNorm2d, 
                      replace_layer_instance=torchvision.ops.FrozenBatchNorm2d)
        self.reduced_channel = nn.Conv2d(2048, embed_dim, kernel_size=1)

    def forward(self, x):
        backbone_out = self.backbone(x)
        reduced_channel = self.reduced_channel(backbone_out)
        return reduced_channel

    def _replace(self, model, layer_to_replace_instance, replace_layer_instance):
        modules = model._modules
        if len(modules) == 0:
            return
        for key, layer in modules.items():
            if isinstance(layer, layer_to_replace_instance):
                cur_layer_features = modules[key].num_features
                modules[key] = replace_layer_instance(num_features=cur_layer_features)
            else:
                self._replace(modules[key],layer_to_replace_instance, replace_layer_instance)

# creating input + posencoding

class JointIPPE(nn.Module):
    def __init__(self, backbone, pos_encoding):
        super().__init__()

        self.backbone = backbone
        self.pos_encoding = pos_encoding

    def forward(self, x):
        backbone_features = self.backbone(x) # [N, C, H, W]
        pos_encoding = self.pos_encoding(backbone_features) # [N, C, H, W]
        pos_backbone_features = backbone_features + pos_encoding
        return pos_backbone_features.flatten(-2), pos_encoding.flatten(-2) # [N, C, HW]