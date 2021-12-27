import os
import torch
import pandas as pd
from glob2 import glob
from src.utils.io import read_image
from torch.utils.data import Dataset
from typing import Callable, Optional, Dict, Tuple

'''
Unsupervised Dogs Dataset from https://www.kaggle.com/michaelfumery/unlabeled-stanford-dags-dataset
'''

class DogsDataset(Dataset):
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        
        if split not in ["train", "test", "val"]:
            print(f"Split {split} not supported. Choose between train and test.")
            quit()

        if not os.path.isdir(root):
            print(f"Dir {root} for Dogs dataset does not exist. Creating dir and downloading data.")
        else:
            print(f"Verifying data dir structure")
            if not self._verify_data(root=root):
                print(f"Data dir {root} does not have the proper structure (train dir, test dir, list_breeds.csv")
                quit()

        self.root = root
        self.transform = transform
        self.split = split if split != "val" else "test"
        self._labels = None
        self.img_paths = [f for f in glob(os.path.join(self.root, self.split, "*.jpg"))]

    def _verify_data(self, root: str) -> bool:
        """verifies Dogs dataset structure

        Arguments:
            root (str): root dir to verify 
        Returns:
            bool: verification bool (True ok, False not ok)
        """

        if os.path.isdir(os.path.join(root, "train")) is False:
            return False
        
        if os.path.isdir(os.path.join(root, "test")) is False:
            return False

        if os.path.exists(os.path.join(root, "list_breeds.csv")) is False:
            return False

        return True

    @property
    def labels(self) -> Dict[str, Tuple[str, int]]:
        """Returns labels with a dictionary structure

        Returns:
            Dict[str, Tuple[str, int]]: labels (e.g. { 'n02085620': ('Chihuahua', 0) })
        """
        
        if self._labels is not None:
            return self._labels

        _df = pd.read_csv(os.path.join(self.root, "list_breeds.csv"), sep=";")
        self._labels = {
            row["Id"]: (row["Breed"], idx) for idx, row in _df.iterrows()
        }
        return self._labels

    def extract_label(self, filename: str) -> Tuple[str, int]:
        """Extract label id given a filename

        Args:
            filename (str): image filename

        Returns:
            Tuple[str, int]: breed name string, breed numerical identifier
        """
        breed_id = filename.split(os.sep)[-1].split("_")[0]
        return self.labels[breed_id]
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, int, str]:
        """
        """
        img_path = self.img_paths[index]
        img = read_image(img_path=img_path)
        if self.transform is not None:
            img = self.transform(img)
        breed, breed_id = self.extract_label(filename=img_path)
        return img, breed_id, breed

    def __len__(self):
        return len(self.img_paths)
        