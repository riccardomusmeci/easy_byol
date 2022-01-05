import os
import numpy as np
import imageio
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose, Normalize

def to_np_img(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))

transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
val_dataset = CIFAR10(
        root='../data', 
        train=False,
        download=False,
        transform=transform
    )

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=1
)

os.makedirs(os.path.join("..", "data", "cifar10_test", "test"), exist_ok=True)

for idx, (img, label) in tqdm(enumerate(val_loader), total=len(val_loader)):
    img = to_np_img(img[0])
    img_name = f"{idx}_{label[0]}.jpg"
    path = f"../data/cifar10_test/test/{img_name}"
    imageio.imwrite(path, img)
    