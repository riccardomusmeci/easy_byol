import torch.nn as nn
from torchvision import models

def get_backbone(model: str = "resnet50") -> nn.Module:
    """Returns specified neural network

    Args:
        model (str, optional): which model. Defaults to "resnet50".

    Returns:
        nn.Module: model
    """

    if model=="resnet50":
        backbone=models.resnet50(pretrained=True)
        layer="avgpool"
    
    if model=="resnet18":
        backbone=models.resnet18(pretrained=True)
        layer="avgpool"
    

    return backbone, layer
        
        