import os
import torch
import datetime
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
from src.model.byol import Encoder
from src.utils.io import save_numpy
from src.utils.io import load_params
from torch.utils.data import DataLoader
from src.dataset.dataset import load_dataset
from src.augmentation.augmentations import get_transform

def save_outputs(folder: str, features: np.array, tsne_features: np.array, labels: np.array):
    """Save outputs from inference

    Args:
        folder (str): folder where saving features and labels
        features (np.array): features
        tsne_features (np.array): TSNE features
        labels (np.array): labels
    """
    # matching output dir with weights name
    
    # Saving features + labels
    save_numpy(
        np_data=features,
        folder=folder,
        filename="features.npy"
    )

    save_numpy(
        np_data=tsne_features,
        folder=folder,
        filename="tsne_features.npy"
    )

    save_numpy(
        np_data=labels, 
        folder=folder, 
        filename="labels.npy"
    )

def inference(args: argparse.Namespace):
    """Performs feature extraction from validation set and saves features+labels

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
    encoder.backbone.load_state_dict(torch.load(args.weights, map_location=torch.device('cpu')))
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
    print("Extracting features from Encoder")
    features, labels = [], []
    for _, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        with torch.no_grad():
            x, _labels = batch
            x = transform(x)
            x = x.to(device)
            _features = encoder(x)
            features.extend(_features.numpy())
            labels.extend(_labels.numpy())

    # Converting to numpy array
    features = np.array(features)
    labels = np.array(labels)

    if args.tsne:
        print("Performing TSNE (this may take a while)")
        tsne = TSNE(n_components=3)
        tsne_features = tsne.fit_transform(X=features, y=labels)
        
    save_outputs(
        folder=os.path.join(args.output_dir, args.weights.split("/")[-2]),
        features=features,
        tsne_features=tsne_features if args.tsne else None,
        labels=labels
    )
            


    
    