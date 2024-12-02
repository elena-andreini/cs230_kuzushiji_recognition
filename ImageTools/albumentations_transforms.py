import albumentations as alb
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image

class AlbumentationsTransform:
    def __init__(self, image_size):
        #self.crop_func = RandomCropAndResize(size=image_size)
        w, h = image_size
        self.aug_func = alb.Compose([
            alb.RGBShift(),
            alb.RandomBrightnessContrast(
                brightness_limit=0.02, 
                contrast_limit=0.02, 
                brightness_by_max=True, 
                p=0.5
            ),
            alb.OneOf([
                alb.Rotate(limit=5),
                alb.GridDistortion(distort_limit=0.2),
                alb.ElasticTransform(alpha=50, sigma=10, alpha_affine=2),
            ], p=0.7),
            alb.OneOf([
                alb.GaussNoise()
                #alb.GaussianNoise()
            ]),
            alb.CoarseDropout(
                max_holes=1, max_height=h // 2, max_width=w // 2,
                min_height=h // 4, min_width=w // 4, fill_value=128),
            ToTensorV2()
        ])

    def __call__(self, img):
        if torch.is_tensor(img):
            img = img.numpy().T
        if isinstance(img, Image.Image):
           img = np.array(img)
        augmented = self.aug_func(image=img)['image']
        return augmented

