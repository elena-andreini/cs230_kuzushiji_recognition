from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from ImageTools.custom_transforms import MinMaxNormalize

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
