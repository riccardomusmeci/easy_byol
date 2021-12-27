import torch.nn as nn
from src.loss.norm_mse import NormalizedMSE

def get_loss_fn(loss: str = "norm_mse") -> nn.Module:
    """returns loss function

    Args:
        loss (str, optional): loss function name. Defaults to "norm_mse".

    Returns:
        nn.Module: loss_fn
    """
    if loss == "norm_mse":
        return NormalizedMSE()
    else:
        print("Only Normalized MSE is implemented. Quitting.")

