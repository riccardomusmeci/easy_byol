import os
import torch
import argparse
from tqdm import tqdm
from src.model.byol import BYOL
from src.loss.loss import get_loss_fn
from torch.utils.data import DataLoader
from src.utils.repr import reproducibility
from src.dataset.dataset import load_dataset
from src.utils.pth_saver import PthSaverLoss
from src.optimizer.optimizer import get_optimizer
from src.optimizer.scheduler import get_scheduler
from src.transform.transform import BYOL_transform
from src.utils.io import load_config, now, copy_config

def train_epoch(loader, model, transform, loss_fn, optimizer, device, log_period = 10):
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
    for idx, batch in enumerate(loader):
        optimizer.zero_grad()
        # batch -> [imgs, labels] or batch -> [imgs, labels idxs, labels name]
        x = batch[0]
        x = x.to(device)
        with torch.no_grad():
            x1, x2 = transform(x), transform(x)
            x1, x2 = x1.to(device), x2.to(device)
        # forward pass
        (pred_1, pred_2), (targ_1, targ_2) = model(x1, x2)
        loss = torch.mean(loss_fn(pred_1, targ_2) + loss_fn(pred_2, targ_1))
        # Backward pass
        loss.backward()
        # Weights update
        optimizer.step()
        # EMA update for the target network
        model.update_target()

        # Logging
        if (idx+1)%log_period == 0:
            print(f"Batch {idx+1}/{len(loader)} - Loss: {loss.item()}")

    del loader

def val_epoch(loader, model, transform, loss_fn, device):
    """validation epoch

    Args:
        loader (DataLoader): validation data loader
        model (nn.Module): model
        transform (Callable): image transformation function
        loss_fn (nn.Module): loss function
        device (str): device (cpu or gpu)
    """

    model.eval()
    loss = 0
    for _, batch in tqdm(enumerate(loader), total=len(loader)):
        with torch.no_grad():
            x = batch[0]
            x = x.to(device)
            x1, x2 = transform(x), transform(x)
            x1, x2 = x1.to(device), x2.to(device)
            (pred_1, pred_2), (targ_1, targ_2) = model(x1, x2)
            loss += torch.mean(loss_fn(pred_1, targ_2) + loss_fn(pred_2, targ_1))

    avg_loss = loss/len(loader)
    print(f"Val loss {avg_loss}")
    del loader
    return avg_loss

def print_setting(config: dict):
    """prints training information

    Args:
        config (dict): training config
    """
    print("Training setting:")
    print(f"\t> Dataset: {config['dataset']}")
    print(f"\t> Model: {config['model_name']}")
    print(f"\t> Model Backbone: {config['model']['backbone']}")
    print(f"\t> Image Size: {config['transform']['img_size']}")
    print(f"\t> Epochs: {config['train']['epochs']}")
    print(f"\t> Optimizer Algorithm: {config['optimizer']['algo']}")
    print(f"\t> Scheduler Algorithm: {config['scheduler']['algo']}")
    print(f"\t> Loss: {config['loss']['type']}")

def train(args: argparse.Namespace):
    """Performs BYOL training

    Args:
        args (argparse.Namespace): arguments
    """

    # reproducibility
    reproducibility(args.seed)
    
    # getting device info
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    # loading config
    config_path = os.path.join(args.config_dir, "byol", "config.yml")
    config = load_config(path=config_path)

    # training output dir
    output_dir = os.path.join(args.checkpoints_dir, "byol", f"byol_{now()}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Model checkpoints will be saved at {output_dir}")
    
    # creating BYOL model
    byol = BYOL(**config["model"])
    
    # to cuda (if possible)
    byol = byol.cuda() if torch.cuda.is_available() else byol
    
    # Image Augmentations
    train_transform = BYOL_transform(
        mode="train", 
        **config["transform"]
    )
    val_transform = BYOL_transform(
        mode="val", 
        img_size=config["transform"]["img_size"]
    )
    
    # loading datasets
    train_dataset, val_dataset = load_dataset(
        name=config["dataset"], 
        img_size=config["transform"]["img_size"]
    )
    
    # getting data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config["train"]["batch_size"]
    )

    # optimizer + scheduler + loss_fn
    optimizer = get_optimizer(
        model=byol,
        **config["optimizer"]
    )
    scheduler_algo = config["scheduler"]["algo"]
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler=scheduler_algo,
        **config["scheduler"][scheduler_algo]
    )
    loss_fn = get_loss_fn(loss=config["loss"]["type"])
    epochs = config["train"]["epochs"]
    
    # saving training config
    copy_config(
        src_config_path=config_path,
        dst_dir=output_dir
    )
    
    # print training setting
    print_setting(config=config)
    
    # init pth saver based on loss
    pth_saver = PthSaverLoss(
        pth_dir=os.path.join(output_dir, "weights"),
        model_name="byol_" + config["model"]["backbone"],
        save_disk=args.save_disk
    )

    print(f"Starting training for {epochs} epochs")
    for epoch in range(epochs):
        print(f"Epoch {epoch}/{epochs}")
        # epoch train
        train_epoch(
            loader=train_loader,
            model=byol,
            transform=train_transform,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            log_period=5,
        )
        scheduler.step()
        
        # validation + checkpoint saving
        if (epoch+1)%args.val_period or True:
            val_loss = val_epoch(
                loader=val_loader,
                model=byol,
                transform=val_transform,
                loss_fn=loss_fn,
                device=device
            )
            
            pth_saver.save(
                model=byol.g.backbone,
                epoch=epoch,
                loss=val_loss
            )
        

        