import os
import yaml
import torch
import datetime 
from typing import Dict

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


def save_model(model, model_dir: str, model_name: str, epoch: int, loss: float):
    """Saves model in a dir

    Args:
        model (nn.Module): model to save
        model_dir (str): where to save model
        epoch (int): training epoch
        loss (float): model loss
    """
    
    os.makedirs(model_dir, exist_ok=True)
    filename = f"{model_name}_epoch_{epoch}_loss{loss:.4f}.pth"
    model_path = os.path.join(model_dir, filename)
    torch.save(model.state_dict(), model_path)
    print(f"Saved model at {filename}.")