import torch
from typing import Tuple
from torchvision.datasets import STL10
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import Dataset

def load_dataset(name: str, mode: str = "train") -> Dataset:
    """Loads dataset

    Args:
        name (str): name of the dataset to load
        mode (str): train/val. Defaults to train.

    Returns:
        Dataset: dataset
    """
    
    if name == "STL10":
        return STL10_dataset(mode=mode)
        
    else:
        print("[ERROR] No other dataset implemented.")
        quit()
    

def STL10_dataset(mode: str = "train") -> Tuple[Dataset, Dataset]:
    """Loading STL10 Dataset

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