import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image


class InfoNCEDataset(Dataset):
    def __init__(self, df, N=1, transform = None, same_transform = None):
      self.data = df
      self.paths = df['char_path'].to_numpy()
      self.labels = df['label'].to_numpy()
      self.N = N
      self.transform = transform
      self.same_transform = same_transform

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
        negative_indices = self.data[self.data['label'] != label].index.to_numpy()
        #print(f'negative indices {negative_indices[:10]}')
        choice = np.random.choice(negative_indices, self.N)
        #handle edge case where there are not enough negatives
        image_collection = []
        image_collection.append(anchor)
        image_collection.append(positive)
        for ni in choice:
            negative = Image.open(self.paths[ni])
            if self.transform:
                negative = self.transform(negative)
            image_collection.append(negative)
            neg_labels.append(self.labels[ni])
        original_labels_collection = []
        original_labels_collection.append(label)
        original_labels_collection.append(label)
        original_labels_collection += neg_labels
        #print(f'collection size {len(original_labels_collection)}')
        return image_collection, original_labels_collection
