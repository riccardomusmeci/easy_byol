import torch
from src.model.byol import BYOL

m = BYOL(
    backbone="resnet50",
    projection_size=256,
    hidden_size=4096
)

img1 = torch.rand((2, 3, 128, 128))
img2 = torch.rand((2, 3, 128, 128))

(pred1, pred2), (targ1, targ2) = m(img1, img2)

print(pred1.shape)
print(pred2.shape)
print(targ1.shape)
print(targ2.shape)




