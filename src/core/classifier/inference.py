import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from src.utils.io import save_numpy
from src.utils.io import load_config
from torch.utils.data import DataLoader
from src.model.classifier import load_classifier
from src.transform.transform import easy_transform  
from src.dataset.dataset import load_dataset, num_classes

def inference(args: argparse.Namespace):
    """Performs feature extraction from validation set and saves features+labels

    Args:
        args (argparse.Namespace): arguments
    """
    
    # getting device info
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on {device}")

    # loading config
    config = load_config(path=args.config)
    print(f"Inference on {config['dataset']} dataset with model {config['model']}.")
    
    # getting num classes of dataset to infer from
    n_classes = num_classes(dataset=config["dataset"])
    
    # Load classifier with weights and setting to eval mode
    model = load_classifier(
        model=config["model"],
        weights_path=args.weights,
        n_classes=n_classes
    )
    model.eval()
    
    # loading transformations
    transform = easy_transform(**config["transform"])
    
    # loading datasets
    _, val_dataset = load_dataset(
        name=config["dataset"], 
        transform=transform,
        img_size=config["transform"]["img_size"]
    )
    
    # getting data loaders
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config["train"]["batch_size"]
    )
    
    print("Starting inference on dataset.")
    correct, total = 0, 0
    for _, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        with torch.no_grad():
            x, labels = batch[0], batch[1]
            x = x.to(device)
            outputs = F.softmax(model(x), dim=0)
            
            # computing predictions and saving metrics
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
                
    acc = 100. * correct // total
    print(f"Test Accuracy {acc:.4f}%")

        
    