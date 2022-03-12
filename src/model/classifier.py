import torch
import torch.nn as nn
from src.model.backbone import get_backbone

def load_classifier(backbone: str,
                    weights_path: str,
                    freeze: bool = True,
                    n_classes: int = 10
                ):
    """Loads classifier model based on ssl backbone

    Args:
        backbone (str): backbone model name (e.g. resnet19)
        weights_path (str): path to backbone weights (trained with ssl byol technique)
        freeze (bool, optional): if True freezes layers of backbone. Defaults to True.
        n_classes (int, optional): number of classification classes. Defaults to 10.
    """
    
    print(f"Getting backbone {backbone}")
    model, _ = get_backbone(
        model=backbone,
        pretrained=False
    )
    
    print(f"Loading backbone weights from {weights_path}")
    byol_pretrained_dict = torch.load(
        weights_path, 
        map_location=torch.device('cpu')
    )
    
    print("Setting weights to backbone")
    for k, v in byol_pretrained_dict.items():
        if k.startswith("fc"):
            continue
        else:
            model.state_dict()[k].copy_(v)
    
    if freeze:
        for name, param in model.named_parameters():
            if name.startswith("fc"):
                continue
            if param.requires_grad:
                param.requires_grad=False
    
    print(f"Setting number of classes to predict to {n_classes}.")
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    return model
    
    
    
    