import os
import yaml
import torch
import datetime 
import numpy as np
from PIL import Image
import torch.nn as nn
from glob2 import glob
from typing import Dict
from src.model.backbone import get_backbone

STRFTIME_FORMAT = "%Y-%m-%d-%H-%M-%S"

def now():
    STRFTIME_FORMAT = "%Y-%m-%d-%H-%M-%S"
    return datetime.datetime.now().strftime(STRFTIME_FORMAT)

def save_params(params: dict, yaml_path: str):
    """Saves dict into yaml file

    Args:
        params (dict): params dict
        path (str): path
    """
    with open(yaml_path, 'w') as f:
        try:
            yaml.safe_dump(params, f)
        except yaml.YAMLError as exc:
            print(exc)
            quit()
        
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

def read_image(img_path: str) -> Image:
    """Keeps reading image until succeed. This can avoid IOErrore incurred by heavy IO process.

    Args:
        img_path (str): path to imag

    Raises:
        IOError: IOError

    Returns:
        Image: PIL Image
    """

    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    img = Image.open(img_path)
      
    return img.convert('RGB')

def save_model(model: nn.Module, model_dir: str, model_name: str, epoch: int, loss: float, save_disk: bool = True):
    """Saves model in a dir

    Args:
        model (nn.Module): model to save
        model_dir (str): where to save model
        model_name (str): self supervised model name
        backbone_name (str): backbone used
        epoch (int): training epoch
        loss (float): model loss val
        save_disk (bool, optional): whether to save only the best model and save disk space. Defaults to True.
        
    """
    best_loss = 100000
    os.makedirs(model_dir, exist_ok=True)
    if save_disk:
        if epoch > 0:
            best_pth = [f for f in glob(os.path.join(model_dir, "*.pth"))][0]
            best_loss = float(best_pth.split(os.sep)[-1].split("_")[-1].split(".pth")[0])
        else:
            best_loss = 10000
    if loss < best_loss:
        if save_disk and epoch > 0:
            print(f"Saving disk space: removing old pth file with worse loss ({best_pth}).")
            os.remove(best_pth)
        filename = f"{model_name}_epoch_{epoch}_loss_{loss:.4f}.pth"
        model_path = os.path.join(model_dir, filename)
        torch.save(model.state_dict(), model_path)
        print(f"Saved backbone at {filename}.")
        
def save_numpy(np_data: np.array, folder: str, filename: str):
    """saves numpy data by ensuring the "npy" ext

    Args:
        np_data (np.array): numpy data
        folder (str): directory
        filename (str): filename
    """
    os.makedirs(folder, exist_ok=True)
    filename = filename if filename.endswith("npy") else filename.replace(".", "_") + ".npy"
    filepath = os.path.join(folder, filename)
    print(f"Saving numpy data at {filepath}")
    with open(filepath, 'wb') as f:
        np.save(f, np_data)
    f.close()