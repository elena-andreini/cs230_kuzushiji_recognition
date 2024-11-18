from torch.utils.data import DataLoader, Dataset
from ImageTools.custom_transforms import MinMaxNormalize

class SimpleImageDataset(Dataset):
    def __init__(self, images, transform):
        self.images = images
        #self.mean = np.mean(images)
        #self.stdev = np.std(images)
        self.trivial_transform = transforms.Compose([
            transforms.ToPILImage(),

            transforms.ToTensor(),
            MinMaxNormalize()])

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            #BinarizeImage(),
            #Enhance(1.5),
            #BlurAndBinarize(),
            #transforms.ToPILImage(),
            #transforms.Resize((28, 28)),  # Resize images to 28x28
            transforms.ToTensor(),
            #AdjustBrightnessContrast(brightness=0.2, contrast=0.2),
            #AddGaussianNoise(mean=0.0, std=0.1),
            #transforms.RandomRotation(5),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomCrop(28, padding=4),
            MinMaxNormalize(),
            ToRGB()
            #transforms.Normalize(self.mean, self.stdev)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        tr_image = self.transform(image)
        #o_image = self.trivial_transform(image)
        return tr_image #, o_image
