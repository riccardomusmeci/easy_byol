import os
import torch
import datetime
import argparse
from tqdm import tqdm
from src.model.byol import Encoder
from src.utils.io import load_params
from torch.utils.data import DataLoader
from src.dataset.dataset import load_dataset
from src.augmentation.augmentations import get_transform

def inference(args: argparse.Namespace):
    """Performs BYOL test

    Args:
        args (argparse.Namespace): arguments
    """

    # getting device info
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on {device}")

    # loading params
    params_path = os.path.join(args.hp_dir, args.model, "hp.yml")
    params = load_params(path=params_path)
    
    # Getting encoder and setting to eval mode
    encoder = Encoder(backbone=params["model"]["backbone"])
    encoder.backbone.load_state_dict(torch.load(args.weights))
    encoder.eval()
    
    # loading datasets
    _, val_dataset = load_dataset(name=params["dataset"], mode="val")
    
    # getting data loaders
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=params["train"]["batch_size"]
    )

    transform = get_transform(
        mode="val", 
        img_size=params["transform"]["img_size"]
    )
    for _, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        with torch.no_grad():
            x, labels = batch
            x = transform(x)
            x = x.to(device)
            features = encoder(x)

    
    