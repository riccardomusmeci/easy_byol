
import os
import torch
from tqdm import tqdm
from argparse import Namespace
import torch.nn.functional as F
from src.loss.loss import get_loss_fn
from torch.utils.data import DataLoader
from src.utils.repr import reproducibility
from src.model.classifier import classifier
from src.utils.pth_saver import PthSaverAccuracy
from src.optimizer.optimizer import get_optimizer
from src.optimizer.scheduler import get_scheduler
from src.transform.transform import easy_transform
from src.utils.io import load_config, now, dump_config
from src.dataset.dataset import load_dataset, num_classes

def print_setting(config: dict):
    """prints training information

    Args:
        config (dict): training config
    """
    print("Training setting:")
    print(f"\t> Dataset: {config['dataset']}")
    print(f"\t> Model: {config['model']}")
    print(f"\t> Backbone Frozen: {config['train']['freeze']}")
    print(f"\t> Image Size: {config['transform']['img_size']}")
    print(f"\t> Epochs: {config['train']['epochs']}")
    print(f"\t> Optimizer Algorithm: {config['optimizer']['algo']}")
    print(f"\t> Scheduler Algorithm: {config['scheduler']['algo']}")
    print(f"\t> Loss: {config['train']['loss']}")

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
    # setting seeed for reproducibility
    reproducibility(args.seed)

    # getting device info
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Training classifier on {device}")

    # loading BYOL config
    byol_config = load_config(path=os.path.join(args.ssl_dir, "config.yml"))
    backbone = byol_config["model"]["backbone"]
    
    # loading Classifier config
    config_path = os.path.join(args.config_dir, "classifier", "config.yml")
    config = load_config(path=config_path)
    config["model"] = backbone

    # training output dir
    output_dir = os.path.join(args.checkpoints_dir, "classifier", f"classifier_{now()}")
    os.makedirs(output_dir)
    print(f"Model checkpoints will be saved at {output_dir}")
    
    n_classes = num_classes(dataset=config['dataset'])
    print(f"Dataset {config['dataset']} - num_classes: {n_classes}")
        
    # creating classifier model
    model = classifier(
        backbone=config["model"],
        weights_path=os.path.join(args.ssl_dir, "weights", args.ssl_pth),
        freeze=config["train"]["freeze"],
        n_classes=n_classes
    )
    
    # to cuda (if possivle)
    model = model.cuda() if torch.cuda.is_available() else model
    
    # loading transformations
    transform = easy_transform(**config["transform"])
    
    # loading datasets
    train_dataset, val_dataset = load_dataset(
        name=config["dataset"], 
        transform=transform,
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
        model=model,
        **config["optimizer"]
    )
    scheduler_algo = config["scheduler"]["algo"]
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler=scheduler_algo,
        **config["scheduler"][scheduler_algo]
    )
    loss_fn = get_loss_fn(loss=config["train"]["loss"])
    epochs = config["train"]["epochs"]
    
    # dumping config dict
    dump_config(
        config=config,
        dst_dir=output_dir
    )
    
    # printing training settings
    print_setting(config)
    
    # init pth saver based on accuracy
    pth_saver = PthSaverAccuracy(
        pth_dir=os.path.join(output_dir, "weights"),
        model_name="classifier_" + config["model"],
        save_disk=args.save_disk
    )
    
    print(f"Starting training for {epochs} epochs")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
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
            
            pth_saver.save(
                model=model,
                epoch=epoch,
                loss=val_loss,
                acc=acc
            )