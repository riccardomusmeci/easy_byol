import torch
from typing import Tuple
from src.dataset.dogs import DogsDataset
from torch.utils.data.dataset import Dataset
from torchvision.datasets import STL10, CIFAR10
from torchvision.transforms import ToTensor, Compose, Normalize

def load_dataset(name: str, mode: str = "train", **kwargs) -> Dataset:
    """Loads dataset

    Args:
        name (str): name of the dataset to load
        mode (str): train/val. Defaults to train.
        **kwargs (dict): other arguments.

    Returns:
        Dataset: dataset
    """
    
    if name == "STL10":
        return STL10_dataset(mode=mode)

    if name == "dogs":
        return dogs_dataset(mode=mode, **kwargs)

    if name == "CIFAR10":
        return cifar10_dataset(mode=mode)
        
    else:
        print("[ERROR] No other dataset implemented.")
        quit()

def dogs_dataset(mode: str = "train", img_size: int = 224) -> Tuple[Dataset, Dataset]:
    """Dogs Dataset loader

    Args:
        mode (str, optional): train/val; with val only validation dataset is returned. Defaults to "train".
        img_size (int, optional): image size for all images in the dataset. Defaults to 224.

    Returns:
        Tuple[Dataset, Dataset]: train + val dataset. If mode==val, train is None
    """
    img_size = (img_size, img_size)
    if mode == "train":
        train_dataset = DogsDataset(
            root="data/dogs", 
            split="train",
            transform=ToTensor()
        )
    else:
        train_dataset = None
    
    val_dataset = DogsDataset(
        root="data/dogs", 
        split="val",
        transform=ToTensor()
    )

    return train_dataset, val_dataset

def STL10_dataset(mode: str = "train") -> Tuple[Dataset, Dataset]:
    """STL10 Dataset loader

    Args:
        mode (str, optional): train/val. with val only validation dataset is returned. Defaults to "train".

    Returns:
        Tuple[Dataset, Dataset]: train + val dataset. If mode==val, train is None
    """
    if mode == "train":
        if torch.cuda.is_available():
            print("[On GPU] Loading more STL10 data: train + unlabeled")
            train_split="train+unlabeled"
        else:
            print("[On CPU] Loading less STL10 data: train")
            train_split="train"
        
        train_dataset = STL10(
            root="data", 
            split=train_split,  # train, train+unlabeled
            download=True, 
            transform=ToTensor()
        )
    else:
        train_dataset = None
    
    val_dataset = STL10(
        root="data", 
        split="test", 
        download=True, 
        transform=ToTensor()
    )

    return train_dataset, val_dataset

def cifar10_dataset(mode: str = "train") -> Tuple[Dataset, Dataset]:
    """CIFAR10 Dataset loader

    Args:
        mode (str, optional): train/val. with val only validation dataset is returned. Defaults to "train".

    Returns:
        Tuple[Dataset, Dataset]: train + val dataset. If mode==val, train is None
    """
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if mode == "train":
        train_dataset = CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=transform
        )
    else:
        train_dataset = None

    val_dataset = CIFAR10(
        root='data', 
        train=False,
        download=True, 
        transform=transform
    )
    
    return train_dataset, val_dataset

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')