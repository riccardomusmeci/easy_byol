import argparse
import os
import torch
import datetime
from tqdm import tqdm
from torch.optim import Adam
from src.model.byol import BYOL
from src.loss.loss import get_loss_fn
from torch.utils.data import DataLoader
from src.dataset.dataset import load_dataset
from src.optimizer.optimizer import get_optimizer
from src.optimizer.scheduler import get_scheduler
from src.utils.io import load_params, save_model, now
from src.augmentation.augmentations import get_transform

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
        # batch -> [imgs, labels]
        x, _ = batch
        x.to(device)
        with torch.no_grad():
            x1, x2 = transform(x), transform(x)
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
            print(f"Batch {idx}/{len(loader)} - Loss: {loss.item()}")

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
    for idx, batch in tqdm(enumerate(loader), total=len(loader)):
        x, _ = batch
        x.to(device)
        x1, x2 = transform(x), transform(x)
        (pred_1, pred_2), (targ_1, targ_2) = model(x1, x2)
        loss += torch.mean(loss_fn(pred_1, targ_2) + loss_fn(pred_2, targ_1))
        if idx == 10:
            break
    
    print(f"Val loss {loss/len(loader)}")
    return loss/len(loader)

def save_training_info(params: dict, output_dir: str):
    """prints training information

    Args:
        params (dict): training params
        output_dir (str): where to save training ingo
    """

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "training_info.txt"), "w") as f:
        f.write(f"Model: {params['model_name']}\n")
        f.write(f"Model Backbone: {params['model']['backbone']}\n")
        f.write(f"Epochs: {params['train']['epochs']}\n")
        f.write(f"Optimizer Algorithm: {params['optimizer']['algo']}\n")
        f.write(f"Scheduler Algorithm: {params['scheduler']['algo']}\n")
        f.write(f"Loss: {params['loss']['type']}\n")
    f.close()

    print("Training information:")
    print(f"\t> Model: {params['model_name']}")
    print(f"\t> Model Backbone: {params['model']['backbone']}")
    print(f"\t> Epochs: {params['train']['epochs']}")
    print(f"\t> Optimizer Algorithm: {params['optimizer']['algo']}")
    print(f"\t> Scheduler Algorithm: {params['scheduler']['algo']}")
    print(f"\t> Loss: {params['loss']['type']}")

def train(args: argparse.Namespace):
    """Performs BYOL training

    Args:
        args (argparse.Namespace): arguments
    """

    # getting device info
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    # loading params
    params_path = os.path.join(args.hp_dir, args.model, "hp.yml")
    params = load_params(path=params_path)

    # training output dir
    output_dir = os.path.join(args.checkpoints_dir, args.model, f"{args.model}_{now()}")
    print(f"Model checkpoints will be saved at {output_dir}")
    
    # creating BYOL model
    byol = BYOL(**params["model"])
    
    # to cuda (if possivle)
    byol = byol.cuda() if torch.cuda.is_available() else byol

    # loading datasets
    train_dataset, val_dataset = load_dataset(name=params["dataset"])
    
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
        model=byol,
        **params["optimizer"]
    )
    scheduler_algo = params["scheduler"]["algo"]
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler=scheduler_algo,
        **params["scheduler"][scheduler_algo]
    )
    loss_fn = get_loss_fn(loss=params["loss"]["type"])
    epochs = params["train"]["epochs"]
    
    # printing and saving training information
    save_training_info(params, output_dir)
    
    print(f"Starting training for {epochs} epochs")
    for epoch in range(epochs):
        print(f"Epoch {epoch}/{epochs}")
        # epoch train
        train_epoch(
            loader=train_loader,
            model=byol,
            transform=get_transform(
                mode="train",
                **params["transform"]
            ),
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
                transform=get_transform(
                    mode="val",
                    img_size=params["transform"]["img_size"]
                ),
                loss_fn=loss_fn
            )
            save_model(
                model=byol,
                model_dir=output_dir,
                model_name=args.model + "_" + params["model"]["backbone"],
                epoch=epoch,
                loss=val_loss
            )