from src.caption_model import * 
import torch 

backbone_layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
cmodel = CaptionModel("dvit_16b", backbone_layers = backbone_layers)
print(cmodel)
a = torch.rand(2,3,256,320)
print(cmodel(a).shape)