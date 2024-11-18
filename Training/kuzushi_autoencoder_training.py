import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#! git clone https://github.com/elena-andreini/cs230_kuzushiji_recognition


sys.path.append('/content/cs230_kuzushiji_recognition')

from ImageTools.custom_transforms import MinMaxNormalize, ToRGB
from DataSets.datasets import SimpleImageDataset
from DataSets.visualization import show_images_autoencoder
from Models.autoencoders import ResNet18Encoder, ResNet18Decoder, ResNet18Autoencoder
from Metrics.metrics import cosine_similarity


from google.colab import drive
drive.mount('/content/drive')

# Load packed data. 
# This dataset is made of all chracters labelled as U+3042
# from the Kaggle Kuzushiji Dataset. Each character image is 
# cropped from the full page image using the ground truth 
# bounding box, it's converted to grayscale and resized to (h, w) = (28, 28)

images = np.load('/content/drive/MyDrive/KuzushijiData/U+3042_packed.npy')

transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            MinMaxNormalize(),
            ToRGB()
])

dataset = SimpleImageDataset(images, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Initialize the model, loss function, and optimizer
model = ResNet18Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Training loop
best_loss = float('inf')
best_model_wts = None
counter = 0
patience = 5
num_epochs = 35
model = model.to(device)
for epoch in range(num_epochs):
    running_loss = 0.0
    for data in dataloader:
        # Forward pass
        data = data.to(device)
        outputs = model(data)
        loss = criterion(outputs, data)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(dataloader)
    # Check if current model is the best so far
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        counter = 0
        best_model_wts = model.state_dict().copy()
    else :
      counter += 1
    if counter > patience:
      break
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the best model weights
torch.save(best_model_wts, '/content/drive/MyDrive/KuzushijiData/best_model.pth')
print(f'Best model saved with loss: {best_loss:.4f}')



#Loading the best model
best_model = ResNet18Autoencoder().to(device)
best_model.load_state_dict(best_model_wts)

# Get a batch of images
dataiter = iter(dataloader)
batch = next(dataiter)

# Use the best model to reconstruct the images
dl_images = batch
best_model.eval()

with torch.no_grad():
  reconstructed = best_model(dl_images.to(device))
  reconstructed = reconstructed.cpu().detach()

show_images_autoencoder(dl_images[30:35], reconstructed[30:35], num_images=5)

# Extract features using the encoder part
encoded_images = []
model.eval()
with torch.no_grad():
    encoded = best_model.encoder(dl_images.to(device)).cpu().detach()
    encoded_images.append(encoded.view(encoded.size(0), -1))

encoded_images = torch.cat(encoded_images, dim=0).numpy()


# Apply t-SNE. The dataset contains 4 historical variations of 
# the character U+3042, hence the 4 components
tsne = TSNE(n_components=4, random_state=42)
tsne_features = tsne.fit_transform(encoded_images)

# Apply K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)  # Adjust the number of clusters as needed
labels = kmeans.fit_predict(tsne_features)

# Visualize the t-SNE result
import matplotlib.pyplot as plt
plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=labels, cmap='viridis')
plt.title("t-SNE with K-means Clustering")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()


from sklearn.cluster import KMeans

# Apply K-means clustering on the t-SNE result
kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust the number of clusters as needed
labels = kmeans.fit_predict(encoded_images)



#Visualizing the activations of the model

# Load a pre-trained model (e.g., ResNet-18)
model = models.resnet18#(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Print the summary
summary(autoencoder.to(device), input_size=(3, 32, 32), device=str(device))

