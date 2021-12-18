import torch
import torch.nn as nn
from src.model.mlp import MLP
from src.model.backbone import get_backbone

# a wrapper class for the encoder neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class Encoder(nn.Module):

    def __init__(self, backbone: str = "resnet50", hidden_size: int = 4096, projection_size: int = 256) -> None:
        """Encoder init

        Args:
            backbone (str, optional): which model to use as backbone. Defaults to "resnet50".
            hidden_size (int, optional): mlp hidden size. Defaults to 4096.
            projection_size (int, optional): mlp projection size. Defaults to 256.
        """
        
        super().__init__()
        
        # Backbone fields
        self.backbone, self.layer = get_backbone(model=backbone)
        
        # Projector fields
        self.projection_size = projection_size
        self.hidden_size = hidden_size

        self._projector = None
        self._projector_dim = None
        self._encoded = torch.empty(0) # contains projector forward pass output
        self._register_hook() # registering hook to get backbone output of interest

    @property
    def projector(self) -> MLP:
        """This is run when you use self.projector in your code

        Returns:
            MLP: projector
        """
        if self._projector is None:
            # print("[projector] Creating projector for the first time")
            self._projector=MLP(
                in_features=self._projector_dim,
                hidden_size=self.hidden_size,
                projection_size=self.projection_size
            )
        return self._projector


    def _hook(self, _, __, output: torch.Tensor):
        """this hook gets the output of the layer of interest, sets _projection_dim if necessary, and the calls @property projector to set self._encoded field

        Args:
            _ (): useless param
            __ (): useless param
            output (torch.Tensor): tensor to flatten    
        """
        # flattening output from layer of interes
        # print(f"[_hook] Output from layer of interest is {output.shape}")
        output = output.flatten(start_dim=1)
        # first run: no _projector_dim was set, so we set it
        if self._projector_dim is None:
            self._projector_dim = output.shape[1]
            # print(f"[_hook] Calling hook for the first time, registering self._projector_dim {self._projector_dim}. Flattened output shape: {output.shape}")
        
        # project the output to get encodings
        # here if self.projector is called the first time, it is created with @property projector
        # print("[_hook] Encoding with projector..")
        self._encoded = self.projector(output)
        # print(f"[_hook] Encoding output {self._encoded.shape}")

    def _register_hook(self):
        """Registers hook for layer of interest
        """
        
        # getting layer of interest
        if isinstance(self.layer, str):
            layer = dict([*self.backbone.named_modules()])[self.layer]
        else:
            layer = list(self.backbone.children())[self.layer]
        
        # print(f"[_register_hook] Registering hook for layer {layer}")
        # registering hook
        layer.register_forward_hook(self._hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone and collecting the encodings from the forward hook

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output encodings
        """
        _ = self.backbone(x) # this calls the forward pass and calls the hook of the layer of interest; we do not need this output
        return self._encoded


        


        
        
        


    
