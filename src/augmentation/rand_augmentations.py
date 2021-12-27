import torch
import random
import torch.nn as nn
from torchvision import transforms as T
from torchvision.transforms.transforms import ColorJitter, GaussianBlur

class RandomColorJitter():

    def __init__(self, brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=.8) -> None:
        """Random Color Jitter Transformation

        Args:
            brightness (float, optional): color jitter brightness value. Defaults to 0.8.
            contrast (float, optional): color jitter contrast value. Defaults to 0.8.
            saturation (float, optional): color jitter saturation value. Defaults to 0.8.
            hue (float, optional): color hue brightness value. Defaults to 0.2.
            p (float, optional): color jitter probability of being called. Defaults to .8.
        """
        self.T = ColorJitter(
            brightness=0.8, 
            contrast=0.8, 
            saturation=0.8, 
            hue=0.2,
        )
        self.p=p

    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        """Call to RandomColorJitter

        Args:
            imgs (torch.Tensor): images to transform

        Returns:
            torch.Tensor: transformed images
        """
        for x in imgs:
            if random.random() < self.p:
                x = self.T(x)

        return imgs


class RandomGaussianBlur():

    def __init__(self, kernel_size=(3, 3), sigma=(.1, 2), p=.1) -> None:
        """Random Gaussian Blur Transformation

        Args:
            kernel (tuple, optional): gaussian jernel size. Defaults to (3, 3).
            sigma (tuple, optional): gausiaan kernel std. Defaults to (.1, 2).
            p (float, optional): gaussian kernel probability of bein called. Defaults to .1.
        """
        self.T = GaussianBlur(
            kernel_size=kernel_size, 
            sigma=sigma
        )
        self.p=p

    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        """Call to Random Gaussian Blur

        Args:
            imgs (torch.Tensor): images to transform

        Returns:
            torch.Tensor: transformed images
        """
        for x in imgs:
            if random.random() < self.p:
                x = self.T(x)

        return imgs