from torchvision.datasets import STL10
from torchvision.transforms import ToTensor

def load_dataset(name: str):

    if name == "STL10":
        train_dataset = STL10(
            root="data", 
            split="train",  # train, train+unlabeled
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
    