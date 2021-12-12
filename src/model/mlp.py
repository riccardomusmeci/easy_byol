import torch.nn as nn
import torch

class MLP(nn.Module):

    def __init__(self, in_features: int, hidden_size: int = 4096, projection_size: int = 256) -> None:
        """MLP implementation

        Args:
            in_features (int): input features size
            hidden_size (int, optional): hidden layer features size. Defaults to 4096.
            projection_size (int, optional): output features size (projection). Defaults to 256.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(
                in_features=in_features, 
                out_features=hidden_size
            ),
            nn.BatchNorm1d(
                num_features=hidden_size
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=hidden_size, 
                out_features=projection_size
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on input tensor x

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return self.model(x)
