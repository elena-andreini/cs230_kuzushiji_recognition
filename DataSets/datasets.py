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


class KuzushijiDualDataset(Dataset):
    def __init__(self, images_dir, annotations_file, char_transform=None, ctx_transform=None, img_size=256, down_ratio=4, fraction = 1.0):
        self.images_dir = images_dir
        self.annotations = pd.read_csv(annotations_file)
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
        #image_path = Path(self.images_dir) / (self.annotations.iloc[idx]['image_id']+'.jpg')
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
