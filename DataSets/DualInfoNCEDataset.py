import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image


class DualInfoNCEDataset(Dataset):
    def __init__(self, df_char_cxt, N=1, transform = None, same_transform = None):
      self.data = df_char_cxt
      self.char_paths = df_char_cxt['char_path'].to_numpy()
      self.cxt_paths = df_char_cxt['ctx_path'].to_numpy()
      self.labels = df_char_cxt['label'].to_numpy()
      self.N = N
      self.transform = transform
      self.same_transform = same_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        char_anchor = Image.open(self.char_paths[idx])
        if self.transform is not None:
          char_anchor = self.transform(char_anchor)
        ctx_anchor = Image.open(self.ctx_paths[idx])
        if self.transform is not None:
          ctx_anchor = self.transform(ctx_anchor)
        
        label = self.labels[idx]
        
        # Find a positive sample (same example with same transformation as anchor)
        pos_idx = idx
        
        char_positive = Image.open(self.char_paths[pos_idx])
        cxt_positive = Image.open(self.cxt_paths[pos_idx])
        
        if self.same_transform is not None:
          char_positive = self.same_transform(char_positive)
          cxt_positive = self.same_transform(cxt_positive)
        
        
                
        # Find a negative sample (different class than anchor)
        neg_idxs = []
        neg_labels = []
        attempts = 0
        negative_indices = self.data[self.data['label'] != label].index.to_numpy()
        #print(f'negative indices {negative_indices[:10]}')
        choice = np.random.choice(negative_indices, self.N)
        #handle edge case where there are not enough negatives
        image_collection = []
        ctx_image_collection  = []
        image_collection.append(char_anchor)
        image_collection.append(char_positive)
        ctx_image_collection.append(ctx_anchor)
        ctx_image_collection.append(cxt_positive)

        for ni in choice:
            char_negative = Image.open(self.char_paths[ni])
            if self.transform:
                char_negative = self.transform(char_negative)
            image_collection.append(char_negative)
            cxt_negative = Image.open(self.cxt_paths[ni])
            if self.transform:
                cxt_negative = self.transform(cxt_negative)
            image_collection.append(char_negative)
            ctx_image_collection.append(cxt_negative)
            neg_labels.append(self.labels[ni])
            
        original_labels_collection = []
        original_labels_collection.append(label)
        original_labels_collection.append(label)
        original_labels_collection += neg_labels
        #print(f'collection size {len(original_labels_collection)}')
        return image_collection, ctx_image_collection, original_labels_collection
