import os
from torch.nn.parameter import Parameter
import yaml
import torch
import torch.nn as nn
import datetime 
from typing import Dict
from src.model.backbone import get_backbone

STRFTIME_FORMAT = "%Y-%m-%d-%H-%M-%S"

def now():
    STRFTIME_FORMAT = "%Y-%m-%d-%H-%M-%S"
    return datetime.datetime.now().strftime(STRFTIME_FORMAT)

def load_params(path: str) -> Dict:
    """Loads YAML file

    Args:
        path (str): path to yml file

    Returns:
        Dict: yaml params
    """
    print(f"Loading parameters from {path}.")
    with open(path, "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            quit()
    return params

def save_model(model: nn.Module, model_dir: str, model_name: str, epoch: int, loss: float):
    """Saves model in a dir

    Args:
        model (nn.Module): model to save
        model_dir (str): where to save model
        model_name (str): self supervised model name
        backbone_name (str): backbone used
        epoch (int): training epoch
        loss (float): model loss
        backbone (bool): whether to save only the backbone part of the model. Defaults to True.
    """
    
    os.makedirs(model_dir, exist_ok=True)
    filename = f"{model_name}_epoch_{epoch}_loss_{loss:.4f}.pth"
    model_path = os.path.join(model_dir, filename)
    torch.save(model.state_dict(), model_path)
    print(f"Saved backbone at {filename}.")
