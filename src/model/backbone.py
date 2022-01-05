import torch.nn as nn
from torchvision import models

def get_backbone(model: str = "resnet50", pretrained: bool = True) -> nn.Module:
    """Returns specified neural network

    Args:
        model (str, optional): which model. Defaults to "resnet50".
        pretrained (bool, optional): pretrained weights. Defaults to True.

    Returns:
        nn.Module: model
    """

    if model not in ["resnet18", "resnet50"]:
        print(f"No backbone supported for model {model}")
        quit()

    if model=="resnet50":
        backbone=models.resnet50(pretrained=pretrained)
        layer="avgpool"
    
    if model=="resnet18":
        backbone=models.resnet18(pretrained=pretrained)
        layer="avgpool"
    
    
    return backbone, layer
        
        