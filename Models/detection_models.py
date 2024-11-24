import torch
from torch import nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import numpy as np

class CenterNet(nn.Module):
    def __init__(self, num_classes=1, backbone_name='resnet18'):
        super(CenterNet, self).__init__()
        self.backbone = models.__dict__[backbone_name](weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-5])
        # Adjust the number of output channels
        feature_dim = 64  # Output feature dimension of resnet18

        self.heatmap_head = nn.Conv2d(feature_dim, num_classes, kernel_size=1)
        self.offset_head = nn.Conv2d(feature_dim, 2, kernel_size=1)
        self.size_head = nn.Conv2d(feature_dim, 2, kernel_size=1)

        #self.heatmap_head = nn.Conv2d(2048, num_classes, kernel_size=1)
        #self.offset_head = nn.Conv2d(2048, 2, kernel_size=1)
        #self.size_head = nn.Conv2d(2048, 2, kernel_size=1)

    def forward(self, x):
        features = self.backbone(x)
        heatmap = torch.sigmoid(self.heatmap_head(features))
        offset = self.offset_head(features)
        size = self.size_head(features)
        return heatmap, offset, size
        
        






def decode_heatmap(heatmap, offset, size, ratio = 4.0, threshold=0.5):
    """
    Expects the heatmap values are between 0 and 1
    """
    #print(f'heatmap shape {heatmap.shape}')
    # List to store bounding boxes for all images in the batch
    all_boxes = []

    # Iterate over the batch dimension
    for batch_index in range(heatmap.shape[0]):
        heatmap_b = heatmap[batch_index].squeeze()
        offset_b = offset[batch_index].squeeze()
        size_b = size[batch_index].squeeze()

        threshold_mask = heatmap_b > threshold

        # Find the coordinates of key points
        y_indices, x_indices = torch.nonzero(threshold_mask, as_tuple=True)

        # Use the offset to adjust positions
        adjusted_x = x_indices.float() + offset_b[0, y_indices, x_indices]
        adjusted_y = y_indices.float() + offset_b[1, y_indices, x_indices]

        # Use the size to determine width and height
        widths = torch.exp(size_b[0, y_indices, x_indices])
        heights = torch.exp(size_b[1, y_indices, x_indices])

        # Create bounding boxes for the current image
        boxes = []
        for i in range(len(adjusted_x)):
            x_center = adjusted_x[i].item()
            y_center = adjusted_y[i].item()
            width = widths[i].item()
            height = heights[i].item()
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            boxes.append([x_min * ratio, y_min * ratio, width * ratio, height * ratio])

        # Append the boxes for the current image to the all_boxes list
        all_boxes.append(boxes)

    return all_boxes




def decode_heatmap_s(heatmap, offset, size, threshold=0.5):
    # Step 1: Apply threshold to heatmap
    heatmap = heatmap.squeeze()  # Remove batch dimension
    offset = offset.squeeze()
    size = size.squeeze()

    heatmap = heatmap.sigmoid()  # Ensure values are between 0 and 1
    threshold_mask = heatmap > threshold

    # Step 2: Find the coordinates of key points
    y_indices, x_indices = torch.nonzero(threshold_mask, as_tuple=True)

    # Step 3: Use the offset to adjust positions
    adjusted_x = x_indices.float() + offset[0, y_indices, x_indices]
    adjusted_y = y_indices.float() + offset[1, y_indices, x_indices]

    # Step 4: Use the size to determine width and height
    widths = torch.exp(size[0, y_indices, x_indices])
    heights = torch.exp(size[1, y_indices, x_indices])
    #widths = size[0, y_indices, x_indices]
    #heights = size[1, y_indices, x_indices]
    # Step 5: Create bounding boxes
    boxes = []
    for i in range(len(adjusted_x)):
        x_center = adjusted_x[i].item()
        y_center = adjusted_y[i].item()
        width = widths[i].item()
        height = heights[i].item()
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        boxes.append([x_min * 4, y_min * 4, width *4, height *4])

    return np.array(boxes)
