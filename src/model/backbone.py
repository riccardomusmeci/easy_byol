import torch.nn as nn
from torchvision import models

def get_backbone(model: str = "resnet50") -> nn.Module:

    if model=="resnet50":
        backbone=models.resnet50(pretrained=True)
        layer="avgpool"
    
    if model=="resnet18":
        backbone=models.resnet18(pretrained=True)
        layer="avgpool"
    

    return backbone, layer
        
        