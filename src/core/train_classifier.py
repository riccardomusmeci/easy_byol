import os
import torch
import random
import numpy as np
from tqdm import tqdm
from argparse import Namespace
import torch.nn.functional as F
from src.loss.loss import get_loss_fn
from torch.utils.data import DataLoader
from src.model.classifier import load_classifier
from src.optimizer.optimizer import get_optimizer
from src.optimizer.scheduler import get_scheduler
from src.transform.transform import easy_transform
from src.dataset.dataset import load_dataset, num_classes
from src.utils.io import load_params, now, save_model, save_params


infer_backbone = lambda x : x.split("/")[-1].split("_")[1]

def reproducibility(seed=42):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_training_info(params: dict, output_dir: str):
    """prints training information

    Args:
        params (dict): training params
        output_dir (str): where to save training ingo
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving training information (txt recap + hp.yml) at {output_dir}")
    save_params(
        params=params,
        dst_path=os.path.join(output_dir, "hp.yml")
    )
    
    print("Training information:")
    print(f"\t> Dataset: {params['dataset']}")
    print(f"\t> Model: {params['model']['backbone']}")
    print(f"\t> Image Size: {params['transform']['img_size']}")
    print(f"\t> Epochs: {params['train']['epochs']}")
    print(f"\t> Optimizer Algorithm: {params['optimizer']['algo']}")
    print(f"\t> Scheduler Algorithm: {params['scheduler']['algo']}")
    print(f"\t> Loss: {params['train']['loss']}")

def train_epoch(loader, model, loss_fn, optimizer, device, log_period = 10):
    """training epoch

    Args:
        loader (DataLoader): training data loader
        model (nn.Module): model with update_target() method
        transform (Callable): image transformation function
        loss_fn (nn.Module): loss function
        optimizer (nn.optimizer): optimizer
        device (str): device (cpu or gpu).
        log_period (int, optional): logging period for loss monitoring. Defaults to 10.
    """
    model.train()
    correct = 0
    total = 0
    for idx, batch in enumerate(loader):
        optimizer.zero_grad()
        # batch -> [imgs, labels] or batch -> [imgs, labels idxs, labels name]
        x, labels = batch[0], batch[1]
        x = x.to(device)
        outputs = F.softmax(model(x), dim=0)
        
        # computing predictions and saving metrics
        _, preds = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        
        # forward pass
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        loss.backward()
        # Weights update
        optimizer.step()

        # Logging
        if (idx+1)%log_period == 0:
            acc = 100. * correct // total
            print(f"Batch {idx+1}/{len(loader)} - Loss: {loss.item()} - Accuracy {acc:.4f}%")

    del loader

def val_epoch(loader, model, loss_fn, device):
    """validation epoch

    Args:
        loader (DataLoader): validation data loader
        model (nn.Module): model
        loss_fn (nn.Module): loss function
        device (str): device (cpu or gpu)
    """
       
    model.eval()
    correct = 0
    total = 0
    loss = 0
    for idx, batch in tqdm(enumerate(loader), total=len(loader)):
        # batch -> [imgs, labels] or batch -> [imgs, labels idxs, labels name]
        with torch.no_grad():
            x, labels = batch[0], batch[1]
            x = x.to(device)
            outputs = model(x)
            # computing predictions and saving metrics
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            # loss computing
            loss += loss_fn(outputs, labels).item()
        
        # Logging
    avg_loss = loss / len(loader)
    acc = 100. * correct / total
    print(f"Validation - Loss: {avg_loss:.4f} - Accuracy {acc:.4f}%")
    del loader
    
    return avg_loss, acc
    
def train(args: Namespace):
    """Performs Classifier training

    Args:
        args (argparse.Namespace): arguments
    """
    
    reproducibility()

    # getting device info
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Training classifier on {device}")

    # loading params
    params_path = os.path.join(args.hp_dir, "classifier", "hp.yml")
    params = load_params(path=params_path)

    # training output dir
    output_dir = os.path.join(args.checkpoints_dir, "classifier", f"classifier_{now()}")
    os.makedirs(output_dir)
    print(f"Model checkpoints will be saved at {output_dir}")
    
    n_classes = num_classes(dataset=params['dataset'])
    print(f"Dataset {params['dataset']} - num_classes: {n_classes}")
    
    # Inferring backbone from BYOL model
    backbone = infer_backbone(args.ssl_weights)
    params["model"]["backbone"] = backbone
        
    # creating classifier model
    model = load_classifier(
        backbone=backbone,
        weights_path=args.ssl_weights,
        freeze=params["model"]["freeze"],
        n_classes=n_classes
    )
    
    # to cuda (if possivle)
    model = model.cuda() if torch.cuda.is_available() else model
    
    # loading transformations
    transform = easy_transform(**params["transform"])
    
    # loading datasets
    train_dataset, val_dataset = load_dataset(
        name=params["dataset"], 
        transform=transform,
        img_size=params["transform"]["img_size"]
    )
    
    # getting data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=params["train"]["batch_size"],
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=params["train"]["batch_size"]
    )

    # optimizer + scheduler + loss_fn
    optimizer = get_optimizer(
        model=model,
        **params["optimizer"]
    )
    scheduler_algo = params["scheduler"]["algo"]
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler=scheduler_algo,
        **params["scheduler"][scheduler_algo]
    )
    loss_fn = get_loss_fn(loss=params["train"]["loss"])
    epochs = params["train"]["epochs"]
    
    # # printing and saving training information
    save_training_info(params, output_dir)
    
    print(f"Starting training for {epochs} epochs")
    for epoch in range(epochs):
        print(f"Epoch {epoch}/{epochs}")
        # epoch train
        train_epoch(
            loader=train_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            log_period=5,
        )
        scheduler.step()
        
        # validation + checkpoint saving
        if (epoch+1)%args.val_period or True:
            val_loss, acc = val_epoch(
                loader=val_loader,
                model=model,
                loss_fn=loss_fn,
                device=device
            )
            save_model(
                model=model, # backbone trained in self supervised manner with BYOL
                model_dir=output_dir,
                model_name="classifier_" + backbone,
                epoch=epoch,
                loss=val_loss,
                acc=acc,
                save_disk=args.save_disk
            )
        