import random
import torch.nn as nn
from torchvision import transforms as T
from src.augmentation.rand_augmentations import RandomColorJitter, RandomGaussianBlur


def get_transform(
    mode: str, 
    img_size: int=96, 
    mean: list = [0.485, 0.456, 0.406], 
    std: list = [0.229, 0.224, 0.225],
    brightness=0.8, 
    contrast=0.8, 
    saturation=0.8, 
    hue=0.2,
    color_jitter_p=.5,
    grayscale_p=.2,
    h_flip_p=.5,
    kernel=(3, 3),
    sigma=(.1, 2),
    gaussian_blur_p=.1,
    ):
    """BYOL Augmentations

    Args:
        mode (str): train, val, test
        img_size (int): image size
        mean (list, optional): Normalization mean. Defaults to [0.485, 0.456, 0.406].
        std (list, optional): Normalization std. Defaults to [0.229, 0.224, 0.225].
        brightness (float, optional): color jitter brightness value. Defaults to 0.8.
        contrast (float, optional): color jitter contrast value. Defaults to 0.8.
        saturation (float, optional): color jitter saturation value. Defaults to 0.8.
        hue (float, optional): color hue brightness value. Defaults to 0.2.
        color_jitter_p (float, optional): color jitter probability of being called. Defaults to .8.
        grayscale_p (float, optional): grayscale transformation probability of being called. Defaults to .2.
        h_flip_p (float, optional): horizontal flip transformation probability of being called. Defaults to .5.
        kernel (tuple, optional): gaussian jernel size. Defaults to (3, 3).
        sigma (tuple, optional): gausiaan kernel std. Defaults to (.1, 2).
        gaussian_blur_p (float, optional): gaussian kernel probability of bein called. Defaults to .1.

    Returns:
        T.Compose: composition of transformations.
    """

    if mode == "train":
        return T.Compose([
            T.Resize(size=img_size),
            T.RandomApply(nn.ModuleList([
                    T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
                ]), 
                p=color_jitter_p
            ),
            T.RandomGrayscale(p=grayscale_p),
            T.RandomHorizontalFlip(p=h_flip_p),
            T.RandomApply(nn.ModuleList([
                    T.GaussianBlur(kernel_size=kernel, sigma=sigma)
                ]),
                p=gaussian_blur_p
            ),
            T.RandomResizedCrop(size=img_size),
            T.Normalize(mean=mean, std=std)
        ])

    if mode in ["test", "val"]:
        return T.Compose([
            T.Resize(size=img_size),
            T.Normalize(mean=mean, std=std)
        ])

    
