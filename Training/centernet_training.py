import matplotlib.pyplot as plt
import torch
import torchvision
import cv2
import numpy as np
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from DataSets.visualization import show_images_and_boxes
from Losses.centernet_losses import centernet_loss

# from google.colab import drive
# drive.mount('/content/drive')


from DataSets.datasets import KuzushijiCenterNetDataset, visualize_centernet_heatmap
from Models.detection_models import CenterNet, decode_heatmap
from DataSets.dataset_preprocessing import copy_dataset
from DataSets.utils import custom_collate_fn


# When working on Google Colab reading data directly from Google Drive is slow
# Copying data to Colab


#images_dir = '/content/kuzushiji-recognition/train_images' #'/content/drive/MyDrive/kuzushiji-recognition/train_images'

# The file with annotations (boxes and unicode points)
annotations_file = '/content/drive/MyDrive/kuzushiji-recognition/train.csv'

# Source directory in Google Drive : location to copy from
src_dir = '/content/drive/MyDrive/kuzushiji-recognition/train_images'
# Destination directory in Colab : location to copy to
dest_dir = '/content/kuzushiji-recognition/train_images'

# Create destination directory if it does not exist
try:
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
except OSError as e:
    print(f"Error creating directory '{dest_dir}': {e}")



# Copying the dataset locally for increased efficiency
copy_dataset(annotations_file, src_dir, dest_dir)


# Leaving out normalization for the moment
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = KuzushijiCenterNetDataset(images_dir = images_dir, annotations_file = annotations_file, transform = transform, fraction = 0.1)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True,  collate_fn=custom_collate_fn)





# Visualize the dataset 
# Get a batch of images
dataiter = iter(dataloader)
batch = next(dataiter)
# Unpack the batch
dl_images, gt_heatmaps, gt_offsets, gt_sizes, boxes = batch
show_images_and_boxes(dl_images, boxes)
plt.show()






# Initialize model, loss function, and optimizer
model = CenterNet(num_classes=1)
criterion = centernet_loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 60
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, gt_heatmaps, gt_offsets, gt_sizes, _ in dataloader:
        optimizer.zero_grad()

        # Forward pass
        pred_heatmaps, pred_offsets, pred_sizes = model(images)

        # Calculate loss
        loss = criterion(pred_heatmaps, pred_offsets, pred_sizes, gt_heatmaps, gt_offsets, gt_sizes)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')

print('Training complete')


# Show some predictions

dataiter = iter(dataloader)
batch = next(dataiter)
images, gt_heatmaps, gt_offsets, gt_sizes, gt_boxes = batch
model.eval()
with torch.no_grad():
  heatmap, offset, size = model(images[0].unsqueeze(0))
  bounding_boxes = decode_heatmap(heatmap, offset, size, threshold=0.67)

print(bounding_boxes)

len(bounding_boxes)

gt_boxes[0][10]

bounding_boxes[0]

gt_boxes[0][3]

images[0].shape

encoded_gt = create_ground_truth_free(gt_boxes[0], images[0].shape, 4)

heatmap_gt, offset_gt, size_gt = encoded_gt

decoded_gt =decode_heatmap(heatmap_gt, offset_gt, size_gt, threshold=0.5)

show_images_1(images[0], bounding_boxes)
plt.show()



# Show the predicted heatmap
image = images[0]  # Dummy image
heatmap = np.random.rand(64, 64)  # Dummy heatmap, assuming downsampled by 4
visualize_centernet_heatmap(image, heatmap, threshold=0.58)

