import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        N = output1.shape[0]
        #euclidean_distance = nn.functional.pairwise_distance(output1.view(N, -1), output2.view(N, -1))
        cosine_similarity = F.cosine_similarity(output1.view(N, -1), output2.view(N, -1), dim=1)
        cosine_distance =  1 - cosine_similarity
        #print(f'euclidean distance shape {euclidean_distance.shape}')
        # loss = torch.mean((1 - label) * torch.pow(cosine_distance, 2) +
        #                   (label) * torch.pow(torch.clamp(self.margin - cosine_distance, min=0.0), 2))
        #loss = torch.mean((label) * cosine_distance +
         #                 (1 -label) * torch.clamp(- self.margin + cosine_similarity, min=0.0))
        # Contrastive loss
        loss_contrastive = torch.mean(
            (label) * cosine_distance  +  # Minimize distance for similar pairs (label = 1)
            (1 - label) * torch.clamp(self.margin - cosine_distance, min=0.0) # Ensure distance is at least margin for dissimilar pairs (label = 0)
        )
        return loss_contrastive


# Updated transforms
transform = transforms.Compose([
    #transforms.RandomResizedCrop(size=64, scale=(0.8, 1.0)),  # Random cropping
    #transforms.RandomRotation(degrees=5),  # Random rotation
    #GridDistortion(num_steps=5, distort_limit=0.3),  # Grid distortion
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Normalize(mean=stats[0], std=stats[1]),
    #transforms.Lambda(lambda x: torch.clamp(x, 0, 1))
    PercentileNormalize(),
    transforms.Lambda(lambda x: to_pil_image(x)),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    #MinMaxNormalize()
    #PercentileNormalize()
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #transforms.Normalize(mean=stats[0], std=stats[1]),
    #transforms.Lambda(lambda x: torch.clamp(x, 0, 1))
    #PercentileNormalize()

])

same_transform = transforms.Compose([
      # Random cropping

    #GridDistortion(num_steps=5, distort_limit=0.3),  # Grid distortion
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Normalize(mean=stats[0], std=stats[1]),
    #transforms.Lambda(lambda x: torch.clamp(x, 0, 1))
    PercentileNormalize(),
    transforms.Lambda(lambda x: to_pil_image(x)),
    transforms.RandomRotation(degrees=4),  # Random rotation
    transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0)),
    #transforms.Resize((128, 128)),
    transforms.ToTensor()
    #MinMaxNormalize()
    #PercentileNormalize()
 
])





