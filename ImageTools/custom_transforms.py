import torch
from torch import tensor

class MinMaxNormalize:
    def __call__(self, tensor):
        min_val = tensor.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        max_val = tensor.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        return normalized_tensor
        
class ToRGB:
  def __call__(self, tensor):
    rgb_tensor = tensor.repeat(3, 1, 1)
    return rgb_tensor
    
    
    
class CustomResize:
    """
    Transform the input image shape into (1, c, w, h)
    and resizes (w,h) to size

    """
    def __init__(self, size):
        self.size = size
        self.resize = transforms.Resize(size)

    def __call__(self, img):
        # Ensure the image is in the correct format
        if len(img.shape) == 2:
            img = img.unsqueeze(0)  # Add channel dimension if not present
        elif len(img.shape) == 3 and img.shape[0] == 1:
            img = img.repeat(3, 1, 1)  # Convert 1-channel to 3-channel
        elif len(img.shape) == 3 and img.shape[0] == 3:
            pass  # Image already has 3 channels
        else:
            raise ValueError("Unsupported image format")
        # Add batch dimension
        img = img.unsqueeze(0)
        # Apply resize
        resized_img = self.resize(img)
        # Remove batch dimension
        resized_img = resized_img.squeeze(0)
        return resized_img