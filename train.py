import torch
from torch import Tensor
import torch.nn.functional as F
from src.model.byol import BYOL
from src.augmentation.augmentations import get_transform
from matplotlib import pyplot as plt
import cv2
from torchvision.utils import save_image

m = BYOL(
    backbone="resnet50",
    projection_size=256,
    hidden_size=4096
)

def normalized_mse(x: Tensor, y: Tensor) -> Tensor:
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return 2 - 2 * (x * y).sum(dim=-1)


# img1 = torch.rand((1, 3, 128, 128))
# img2 = torch.rand((1, 3, 128, 128))

# m.eval()
# (pred1, pred2), (targ1, targ2) = m(img1, img2)
# # (pred1, pred2) = m(img1, img2)

# print(pred1.shape)
# print(pred2.shape)
# print(targ1.shape)
# print(targ2.shape)

from torchvision.datasets import STL10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


# TRAIN_DATASET = STL10(root="data", split="train", download=True, transform=ToTensor())
TRAIN_UNLABELED_DATASET = STL10(
    root="data", split="train+unlabeled", download=True, transform=ToTensor()
)
TEST_DATASET = STL10(root="data", split="test", download=True, transform=ToTensor())


# train_loader = DataLoader(
#     dataset=TRAIN_DATASET,
#     batch_size=16,
#     shuffle=True,
#     drop_last=True
# )

train_unsup_loader = DataLoader(
    dataset=TRAIN_UNLABELED_DATASET,
    batch_size=16,
    shuffle=True,
    drop_last=True
)


epochs = 10
# loss =

transform = get_transform(mode="train", img_size=96)

for epoch in range(epochs):
    for batch in train_unsup_loader:
        # batch is a list of two Tensor:
        # - batch[0] -> img
        # - batch[1] -> labels
        # we are interested in batch[0] since we are in self-supervised mode
        x = batch[0]
        # Generating view
        with torch.no_grad():
            x1, x2 = transform(x), transform(x)
        
        (pred_1, pred_2), (targ_1, targ_2) = m(x1, x2)

        loss = torch.mean(normalized_mse(pred_1, targ_2) + normalized_mse(pred_2, targ_1))
        print(f"Loss: {loss}")
        quit()
        





# for batch in train_loader:
#     print(batch)
#     break


# val_loader = DataLoader(
#     dataset=TEST_DATASET,
#     batch_size=128,
# )
