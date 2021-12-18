from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler

def get_scheduler(optimizer: Optimizer, scheduler: str, **kwargs) -> _LRScheduler:
    """returns lr scheduler

    Args:
        optimizer (Optimizer): optimizer
        scheduler (str): which scheduler

    Returns:
        _LRScheduler: scheduler
    """
    
    if scheduler == "cosine":
        return CosineAnnealingLR(
            optimizer=optimizer,
            **kwargs
        )
    else:
        print("No other scheduler implemented")
        quit()