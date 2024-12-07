import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from ImageTools.custom_transforms import MinMaxNormalize
from ImageTools.utils import calculate_padding,edge_aware_pad
import ImageTools

class SimpleImageDataset(Dataset):
    def __init__(self, images, transform= None):
        self.images = images
        #self.mean = np.mean(images)
        #self.stdev = np.std(images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image

class KuzushijiSingleDataset(Dataset):
    def __init__(self, df, transform=None):
        self.annotations = df
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        label = row['label']
        char_image_path = row['char_path']
        char_image = cv2.imread(str(char_image_path))
        char_padding = calculate_padding(char_image)
        char_image = edge_aware_pad(char_image, char_padding)
        if self.transform:
            char_image = self.transform(char_image)
        return char_image, label



def draw_gaussian(heatmap, center, sigma=2):
    """
    Draw a 2D Gaussian on the heatmap centered at 'center'.

    :param heatmap: The heatmap to draw the Gaussian on (numpy array).
    :param center: The center of the Gaussian (x, y coordinates).
    :param sigma: The standard deviation of the Gaussian.
    :return: The heatmap with the Gaussian drawn on it.
    """
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[1], heatmap.shape[0]

    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    if ul[0] >= w or ul[1] >= h or br[0] < 0 or br[1] < 0:
        return heatmap

    g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], h) - ul[1]
    img_x = max(0, ul[0]), min(br[0], w)
    img_y = max(0, ul[1]), min(br[1], h)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    )
    return heatmap




def create_heatmap_with_gaussian(bboxes, img_shape, down_ratio=4, sigma=2):
    output_shape = (img_shape[1] // down_ratio, img_shape[2] // down_ratio)
    heatmap = np.zeros((1, output_shape[0], output_shape[1]), dtype=np.float32)

    for bbox in bboxes:
        x, y, w, h = bbox
        center_x, center_y = (x + w / 2) // down_ratio, (y + h / 2) // down_ratio
        heatmap[0] = draw_gaussian(heatmap[0], (center_x, center_y), sigma=sigma)

    return heatmap
    
def smooth_tensors_with_gaussian(centers, heatmap, offset, size, sigma=2):
    for center in centers:
        draw_gaussian(heatmap, center, sigma)
        draw_gaussian(offset[0], center, sigma)
        draw_gaussian(offset[1], center, sigma)
        draw_gaussian(size[0], center, sigma)
        draw_gaussian(size[1], center, sigma)



class KuzushijiDualDataset(Dataset):
    def __init__(self, annotations_df, char_transform=None, ctx_transform=None, img_size=256, down_ratio=4, fraction = 1.0):
        self.annotations = annotations_df
        self.char_transform = char_transform
        self.ctx_transform = ctx_transform
        self.img_size = img_size
        self.down_ratio = down_ratio
        # Reduce the dataset to a fraction
        if fraction < 1.0:
            self.annotations = self.annotations.sample(frac=fraction).reset_index(drop=True)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        label = row['label']
        char_image_path = row['char_path']
        ctx_image_path = row['ctx_path']
        char_image = cv2.imread(str(char_image_path))
        ctx_image = cv2.imread(str(ctx_image_path))
        char_padding = ImageTools.utils.calculate_padding(char_image)
        ctx_padding = ImageTools.utils.calculate_padding(ctx_image)
        char_image = ImageTools.utils.edge_aware_pad(char_image, char_padding)
        ctx_image = ImageTools.utils.edge_aware_pad(ctx_image, ctx_padding)
        if self.char_transform:
            char_image = self.char_transform(char_image)
        if self.ctx_transform:
            ctx_image = self.ctx_transform(ctx_image)

        return char_image, ctx_image, label



class KuzushijiCenterNetDataset(Dataset):
    def __init__(self, images_dir, annotations_file, transform=None, img_size=256, down_ratio=4, fraction = 1.0):
        self.images_dir = images_dir
        self.annotations = pd.read_csv(annotations_file)
        self.annotations = self.annotations
        self.transform = transform
        self.img_size = img_size
        self.down_ratio = down_ratio
        # Reduce the dataset to a fraction
        if fraction < 1.0:
            self.annotations = self.annotations.sample(frac=fraction).reset_index(drop=True)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_path = Path(self.images_dir) / (self.annotations.iloc[idx]['image_id']+'.jpg')
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        padding = calculate_padding(image)
        image = edge_aware_pad(image, padding)
        padded_shape = image.shape
        if self.transform:
            image = self.transform(image)
        transformed_shape = image.shape
        scale_factor = transformed_shape[1]/padded_shape[1]
        bboxes = self.parse_annotations(self.annotations.iloc[idx]['labels'])
        for i in range(len(bboxes)):
          bboxes[i] = ( (bboxes[i][0] + padding[0]) * scale_factor, ( bboxes[i][1] + padding[1]) * scale_factor,  bboxes[i][2] * scale_factor,  bboxes[i][3]*scale_factor)
        gt_heatmap, gt_offset, gt_size = self.create_ground_truth(bboxes, image.shape)
        return image, gt_heatmap, gt_offset, gt_size, bboxes

    def parse_annotations(self, label_string):
        bboxes = []
        labels = label_string.split()
        for i in range(0, len(labels), 5):
            unicode_char = labels[i]
            x = int(labels[i + 1])
            y = int(labels[i + 2])
            w = int(labels[i + 3])
            h = int(labels[i + 4])
            bboxes.append((x, y, w, h))
        return bboxes

    def create_ground_truth(self, bboxes, img_shape, epsilon = 1e-6):
        heatmap = np.zeros((1, img_shape[1] // self.down_ratio, img_shape[2] // self.down_ratio))
        offset = np.zeros((2, img_shape[1] // self.down_ratio, img_shape[2] // self.down_ratio))
        size = np.zeros((2, img_shape[1] // self.down_ratio, img_shape[2] // self.down_ratio))
        #heatmap = create_heatmap_with_gaussian(bboxes, img_shape, self.down_ratio)
        centers = []
        for bbox in bboxes:
            x, y, w, h = bbox
            center_x, center_y = int(x + w / 2) // self.down_ratio, int(y + h / 2) // self.down_ratio
            centers.append((center_x, center_y))
            heatmap[0, center_y, center_x] = 1
            offset[0, center_y, center_x] = (x + w / 2) / self.down_ratio - center_x
            offset[1, center_y, center_x] = (y + h / 2) / self.down_ratio - center_y
            #size[0, center_y, center_x] = np.log(max(w / self.down_ratio, epsilon))
            #size[1, center_y, center_x] = np.log(max(h / self.down_ratio, epsilon))
            size[0, center_y, center_x] = w / self.down_ratio
            size[1, center_y, center_x] = h / self.down_ratio
            
        #smooth_tensors_with_gaussian(centers, heatmap, offset, size)
        
        return torch.tensor(heatmap, dtype=torch.float32), torch.tensor(offset, dtype=torch.float32), torch.tensor(size, dtype=torch.float32)
        
        
def create_centernet_ground_truth(bboxes, img_shape, down_ratio, epsilon = 1e-6):
        heatmap = np.zeros((1, img_shape[1] // down_ratio, img_shape[2] // down_ratio))
        offset = np.zeros((2, img_shape[1] // down_ratio, img_shape[2] // down_ratio))
        size = np.zeros((2, img_shape[1] // down_ratio, img_shape[2] // down_ratio))
        #heatmap = create_heatmap_with_gaussian(bboxes, img_shape, down_ratio)
        centers = []
        for bbox in bboxes:
            x, y, w, h = bbox
            center_x, center_y = int(x + w / 2) // down_ratio, int(y + h / 2) // down_ratio
            centers.append((center_x, center_y))
            heatmap[0, center_y, center_x] = 1
            offset[0, center_y, center_x] = (x + w / 2) / down_ratio - center_x
            offset[1, center_y, center_x] = (y + h / 2) / down_ratio - center_y
            #size[0, center_y, center_x] = np.log(max(w / self.down_ratio, epsilon))
            #size[1, center_y, center_x] = np.log(max(h / self.down_ratio, epsilon))
            size[0, center_y, center_x] = w / down_ratio
            size[1, center_y, center_x] = h / down_ratio
            
        #smooth_tensors_with_gaussian(centers, heatmap, offset, size)
        
        return torch.tensor(heatmap, dtype=torch.float32), torch.tensor(offset, dtype=torch.float32), torch.tensor(size, dtype=torch.float32)
        
        
def visualize_centernet_heatmap(image, heatmap, threshold=0.5):
    """
    Visualize the heatmap overlaid on the original image.

    :param image: The original image (numpy array).
    :param heatmap: The heatmap from the model (numpy array).
    :param threshold: Threshold value to binarize the heatmap.
    """
    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()

    # Apply threshold
    thresholded_heatmap = (heatmap > threshold).astype(np.float32)
    # Convert image to channel-last format [H, W, C]
    image = np.transpose(image, (1, 2, 0))
    # Convert image to RGB if it's grayscale
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)

    # Resize heatmap to match image size if needed
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Visualize heatmap
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.imshow(heatmap, cmap='jet', alpha=0.5)  # Overlay heatmap with some transparency
    plt.colorbar()
    plt.show()
