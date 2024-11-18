import sys
import shutil
import os
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0 as TVEfficientNet
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import random





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


sys.path.append('/content/cs230_kuzushiji_recognition')

# Commented out IPython magic to ensure Python compatibility.
# %load_ext autoreload
# %autoreload 2

from Models.classification_models import SelfAttention, ContextAwareModel, KuzushijiDualModel
from Losses.custom_losses import ContrastiveLoss
from DataSets import dataset_eda, dataset_preprocessing
from DataSets.visualization import show_images_1
from ImageTools import custom_transforms as ctr
from  ImageTools import utils
import ImageTools
from Metrics.metrics import calculate_accuracy

images_dir = '/content/kuzushiji-recognition/train_images' 
annotations_file = '/content/drive/MyDrive/kuzushiji-recognition/train.csv'

# from google.colab import drive
# drive.mount('/content/drive')

# ! mkdir /content/kuzushiji-recognition/train_images
# ! mkdir /content/kuzushiji-recognition/train_images_ctx


# Source and destination directories
src_dir = '/content/drive/MyDrive/kuzushiji-recognition/train_images'
local_src_dir = '/content/kuzushiji-recognition/train_full_images'

# Create the destination directory if it doesn't exist
os.makedirs(local_src_dir, exist_ok=True)
df = pd.read_csv(annotations_file)

# Copying files locally
# for file_name in df['image_id'][:100]:
    # src_path = os.path.join(src_dir, file_name+'.jpg')
    # dest_path = os.path.join(local_src_dir, file_name+'.jpg')
    # shutil.copy(src_path, dest_path)



# Get the 5 most frequent classes
stats = dataset_eda.get_kuzushiji_stats(annotations_file)
top_classes = dict(stats[:5])
labels = dataset_eda.convert_labels(top_classes)


# Read annotations
df = pd.read_csv(annotations_file)

#! rm -r /content/kuzushiji-recognition/train_images_ctx/


#! mkdir  /content/kuzushiji-recognition/train_images/'


char_images_dir = '/content/kuzushiji-recognition/train_images/'
ctx_images_dir = '/content/kuzushiji-recognition/train_images_ctx/'
proc_annotations_dir = '/content/kuzushiji-recognition/'
train_annotations_file = '/content/kuzushiji-recognition/train_labelled_data.csv'
valid_annotations_file = '/content/kuzushiji-recognition/valid_labelled_data.csv'

# Create directories for the processed dataset if they do not exist
if not os.path.exists(char_images_dir): 
    os.makedirs(char_images_dir)
    
if not os.path.exists(ctx_images_dir): 
    os.makedirs(ctx_images_dir)

# Generate the dataset
dataset.preprocessing.generate_classification_dataset(df, labels, local_src_dir, char_images_dir, ctx_images_dir, proc_annotations_dir)

# Split the dataset in test and validation
dataset.preprocessing.split_classification_dataset(proc_annotations_dir, train_annotations_file, valid_annotations_file)


# Feature extractor for individual characters
char_extractor = TVEfficientNet(pretrained = True).to(device)
char_extractor = nn.Sequential(*list(char_extractor.children())[:-1])

# Feature extractor for context-aware patches
context_extractor = TVEfficientNet(pretrained=True).to(device)
context_extractor = nn.Sequential(*list(context_extractor.children())[:-1])


# Defining transformations for the character and the context branch

char_transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize((128, 128)),
      transforms.ToTensor(),
      ImageTools.custom_transforms.MinMaxNormalize(),
  ])

ctx_transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize((128, 128)),
      transforms.ToTensor(),
      ImageTools.custom_transforms.MinMaxNormalize(),
  ])

# Create the dataset and dataloader

train_dataset = KuzushijiDualDataset(annotations_file=train_annotations_file, char_transform=char_transform, ctx_transform=ctx_transform)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_dataset = KuzushijiDualDataset(annotations_file=valid_annotations_file, char_transform=char_transform, ctx_transform=ctx_transform)
valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=True)


# For the moment trying the the 5 most frequent classes
num_classes = 5

context_model = ContextAwareModel(context_extractor).to(device)
criterion = nn.CrossEntropyLoss()
model = KuzushijiDualModel(char_extractor, context_model, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# Training loop
def train_dual_branch(model, train_dataloader, valid_dataloader, criterion, optimizer, epochs=10):
    model = model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for char_input, context_input, label in train_dataloader:
            optimizer.zero_grad()
            integer_labels = [klabels[l] for l in label]
            integer_labels = torch.tensor(integer_labels).to(device)
            char_input = char_input.to(device)
            context_input = context_input.to(device)
            output = model(char_input, context_input)

            loss = criterion(output, integer_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_dataloader)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0
        with torch.no_grad():
            for char_input, context_input, label in valid_dataloader:
                integer_labels = [klabels[l] for l in label]
                integer_labels = torch.tensor(integer_labels).to(device)
                char_input = char_input.to(device)
                context_input = context_input.to(device)
                output = model(char_input, context_input)
                loss = criterion(output, integer_labels)
                val_loss += loss.item()
                val_correct += calculate_accuracy(output, integer_labels) * char_input.size(0)
                val_samples += char_input.size(0)
        val_accuracy = val_correct / val_samples
        avg_val_loss = val_loss / len(valid_dataloader)
        model.train()  # Set model back to training mode
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f} Validation Accuracy: {val_accuracy:.4f}')
    print("Training completed!")

train_dual_branch(model, train_dataloader, valid_dataloader, criterion, optimizer, epochs=35)


# Visualize some predictions
viter = iter(valid_dataloader)
batch = next(viter)
char_imgs, ctxt_imgs, labels = batch

model.eval()
with torch.no_grad():
  preds = model(char_imgs.to(device), ctxt_imgs.to(device))

preds = torch.argmax(preds, dim=1)
preds[:5]
