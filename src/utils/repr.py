import torch
import random
import numpy as np


def reproducibility(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
