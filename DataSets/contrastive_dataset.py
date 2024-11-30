class ContrastiveDataset(Dataset):
    def __init__(self, annotations_file, transform=None,  same_transform = None):
        self.annotations = pd.read_csv(annotations_file)
        self.transform = transform
        self.same_transform = same_transform
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img1_path = self.annotations.iloc[idx]['char_path']
        label1 = self.annotations.iloc[idx]['label']

        # Choose a positive or negative pair randomly
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            # Positive pair (same class)
            #img2_idx = self.annotations[self.annotations['label'] == label1].sample().index[0]
            img2_idx = idx
        else:
            # Negative pair (different class)
            img2_idx = self.annotations[self.annotations['label'] != label1].sample().index[0]

        img2_path = self.annotations.iloc[img2_idx]['char_path']
        label2 = self.annotations.iloc[img2_idx]['label']

        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        if self.transform:
            img1 = self.transform(img1)
        if should_get_same_class:
          if self.same_transform:
            img2 = self.same_transform(img2)
        else:
          if self.transform:
            img2 = self.transform(img2)

        # Return image pairs and a label indicating if they are from the same class (1) or not (0)
        same_class = torch.tensor([1 if label1 == label2 else 0], dtype=torch.float32)

        return (img1, img2), same_class, (label1, label2)


