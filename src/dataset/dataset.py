import torch
from torchvision.datasets import STL10
from torchvision.transforms import ToTensor

def load_dataset(name: str):
    """Loads dataset

    Args:
        name (str): name of the dataset to load

    Returns:
        Dataset: dataset
    """
    
    if name == "STL10":
        if torch.cuda.is_available():
            print("On GPU: loading more STL10 data: train + unlabeled")
            train_split="train+unlabeled"
        else:
            print("On GPU: loading less STL10 data: train")
            train_split="train"
        train_dataset = STL10(
            root="data", 
            split=train_split,  # train, train+unlabeled
            download=True, 
            transform=ToTensor()
        )
        val_dataset = STL10(
            root="data", 
            split="test", 
            download=True, 
            transform=ToTensor()
        )
    else:
        print("[ERROR] No other dataset implemented.")
        quit()
    
    return train_dataset, val_dataset
    