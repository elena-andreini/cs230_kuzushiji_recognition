import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image


class InfoNCEDataset(Dataset):
    def __init__(self, df, N=1, transform = None, same_transform = None):
      self.data = df
      self.paths = df['char_path'].to_numpy()
      self.labels = df['label'].to_numpy()
      self.N = N

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor = Image.open(self.paths[idx])
        if self.transform is not None:
          anchor = self.transform(anchor)
        label = self.labels[idx]
        
        # Find a positive sample (same example with same transformation as anchor)
        pos_idx = idx
        
        positive = Image.open(self.paths[pos_idx])
        if self.same_transform is not None:
          positive = self.same_transform(positive)
        
        # Find a negative sample (different class than anchor)
        neg_idxs = []
        neg_labels = []
        attempts = 0
        while len(neg_idxs) < N:
            neg_idx = idx
            while neg_idx == idx:
                neg_idx = torch.randint(0, len(self.data), (1,)).item()
                attempts += 1
                if attempts > 3 * self.N :
                    break
                if self.labels[neg_idx] != label:
                    neg_idxs.append(neg_idx)
                    neg_labels.append(self.labels[neg_idx])
                    break
        while len(neg_idxs) < self.N and len(neg_idxs) > 0:
            neg_idxs.append(neg_idxs[-1])
            neg_labels.append(neg_labels[-1])
            
        image_collection = []
        image_collection.append(anchor)
        image_collection.append(positive)
        for ni in neg_idxs:
            negative = Image.open(self.paths[ni])
            if self.transform:
                negative = self.transform(negative)
            image_collection.append(negative)
        
        original_labels_collection = []
        original_collection.append(label)
        original_collection.append(label)
        original_collection += neg_labels
        
        return image_collection, original_labels_collection
