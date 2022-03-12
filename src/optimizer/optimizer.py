import torch.nn as nn
from torch.optim import Optimizer, Adam, SGD

def get_optimizer(model: nn.Module, algo: str = "adam", lr: float = 1e-4, weight_decay: float = 1e-6) -> Optimizer:
    """returns the optimizer to train the model

    Args:
        model (nn.Module): model
        algo (str, optional): optimizer algorithm. Defaults to "Adam".
        lr (float, optional): learning rate. Defaults to 1e-4.
        weight_decay (float, optional): weighe decay. Defaults to 1e-6.

    Returns:
        Optimizer: optimizer
    """
    if algo == "adam":
        return Adam(
            params=model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        print("Only adam is implemented as optimizer. Quitting.")
        quit()


