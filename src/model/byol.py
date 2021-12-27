import torch
import torch.nn as nn
from typing import Tuple
from src.model.mlp import MLP
from src.model.encoder import Encoder

class BYOL(nn.Module):

    def __init__(self, backbone: str, hidden_size: int = 4096, projection_size: int = 256, beta: float = 0.999) -> None:
        super().__init__()

        # defining encoder + predictor
        # g_theta in the paper
        self.g = Encoder(
            backbone=backbone,
            hidden_size=hidden_size,
            projection_size=projection_size
        )
        # q_theta in the paper
        self.q = MLP(
            in_features=projection_size,
            hidden_size=hidden_size,
            projection_size=projection_size

        )
        # f_eps in the paper
        self.f = Encoder(
            backbone=backbone,
            hidden_size=hidden_size,
            projection_size=projection_size
        )
        # initializes self.target with self.encoder weights
        self._init_target_weights()

        # EMA weights
        self.beta = beta

    @torch.no_grad()
    def _init_target_weights(self):
        """inits f() weights with g() weights. Also sets the gradient of f() to False.
        """
        for params_g, params_f in zip(self.g.parameters(), self.f.parameters()):
            params_f.data.copy_(params_g.data) # copying g params
            params_f.requires_grad = False # no grad updates 

    @torch.no_grad()
    def update_target(self):
        """EMA update of the target network
        """
        for params_g, params_f in zip(self.g.parameters(), self.f.parameters()):
            params_f.data = self.beta * params_f.data + (1 - self.beta) * params_g.data


    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple:
        """Runs a forward pass for BYOL

        Args:
            x1 (torch.Tensor): first image view
            x2 (torch.Tensor): second image view

        Returns:
            Tuple: (prediction_1, prediction_2), (target_1, target_2)
        """
        pred1, pred2 = self.q(self.g(x1)), self.q(self.g(x2))
        with torch.no_grad():
            targ1, targ2 = self.f(x1), self.f(x2)
        return (pred1, pred2), (targ1, targ2)

    





