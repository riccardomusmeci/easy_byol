import torch
import numpy as np
from torch.nn import functional as F


class GradCAM():
    
    def __init__(self, model, target_layer: str, relu=True, device="cuda"):
        
        self.device = device
        self.model = model

        # a set of hook function handlers
        self.handlers = []

        self.fmap_pool = {}
        self.grad_pool = {}
        self.target_layer = target_layer
        self.relu = relu

        # defining hooks to generate GradCAM
        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def remove_hook(self):
        for handle in self.handlers:
            handle.remove()

    def forward(self, x):
        self.image_shape = x.shape[2:]
        self.logits = self.model(x)
        return self.logits

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            for n, p in self.model.named_modules():
                print(n)
            raise ValueError("Invalid layer name: {}".format(target_layer))
            
    def __call__(self, input_tensor: torch.Tensor, target_category=None):
        
        self.model.eval()
        target_category=target_category

        with torch.set_grad_enabled(True):
            scores = self.forward(input_tensor)
            _, out_arg_max = torch.max(scores, 1)
        
        self.backward(ids=out_arg_max.unsqueeze(1))
        grad_cam_regions = self._generate()

        return grad_cam_regions

    def _generate(self) -> np.array:
        """Generates GradCAM heatmap

        Returns:
            np.array: GradCAM heatmaps for batch images
        """

        fmaps = self._find(self.fmap_pool, self.target_layer)
        grads = self._find(self.grad_pool, self.target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        if self.relu:
            gcam = F.relu(gcam)

        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        min_map = gcam.min(dim=1, keepdim=True)[0]
        max_map = gcam.max(dim=1, keepdim=True)[0]
        gcam -= min_map
        gcam /= max_map - min_map + 1e-12
        gcam = gcam.view(B, C, H, W)

        gcam = gcam.permute(0, 2, 3, 1).numpy()

        return gcam