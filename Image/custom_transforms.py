import torch
from torch import tensor

class MinMaxNormalize:
    def __call__(self, tensor):
        min_val = tensor.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        max_val = tensor.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        return normalized_tensor