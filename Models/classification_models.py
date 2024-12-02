import torch
import torch.nn as nn
import torch.nn.functional as F



class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, width, height)
        out = out + x
        return out

# Adding self-attention to the context extractor
class ContextAwareModel(nn.Module):
    def __init__(self, base_model):
        super(ContextAwareModel, self).__init__()
        self.base_model = base_model
        self.attention = SelfAttention(in_dim=1280)  # Adjust in_dim based on your model's output channels

    def forward(self, x):
        features = self.base_model(x)
        attended_features = self.attention(features)
        return attended_features

class SelfMHAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfMHAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.proj_value = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        # Reshape from (B, C, W, H) to (W*H, B, C)
        x = x.view(batch_size, channels, -1).permute(2, 0, 1)
        # MultiheadAttention forward pass
        attn_output, _ = self.multihead_attn(x, x, x)
        # Reshape back to (B, C, W, H)
        attn_output = attn_output.permute(1, 2, 0).view(batch_size, channels, width, height)
        # Apply projection and residual connection
        out = self.proj_value(attn_output) + x.permute(1, 2, 0).view(batch_size, channels, width, height)
        return out
        
        
class ContextAwareModel2(nn.Module):
    def __init__(self, base_model, embed_dim, num_heads):
        super(ContextAwareModel2, self).__init__()
        self.base_model = base_model
        self.attention = SelfMHAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, x):
        features = self.base_model(x)
        attended_features = self.attention(features)
        return attended_features





class KuzushijiDualModel(nn.Module):
    def __init__(self, char_model, context_model, num_classes):
        super(KuzushijiDualModel, self).__init__()
        self.char_model = char_model
        self.context_model = context_model
        self.fc = nn.Linear(2560, 128)  # Combine features from both branches
        #self.fc = nn.Linear(1280, 128)  # When using a single branch
        self.classifier = nn.Linear(128, num_classes)  # Final classification layer

    def forward(self, char_input, context_input, return_embeddings = False):
        char_features = self.char_model(char_input).view(char_input.size(0), -1)
        context_features = self.context_model(context_input).view(context_input.size(0), -1)
        combined_features = torch.cat((char_features, context_features), dim=1)
        combined_features = self.fc(combined_features)
        combined_features = F.relu(combined_features)  # Apply ReLU activation
        #char_features = self.fc(char_features) # code variation for single branch
        #char_features = F.relu(char_features) #code variation for single branch
        logits = self.classifier(combined_features)
        output = F.softmax(logits, dim=1)  # Apply softmax to get probabilities
        if return_embeddings:
            return combined_features
        else:
            return output


class KuzushijiSimpleClassificationModel(nn.Module):
    def __init__(self, char_model, num_classes):
        super(KuzushijiDualModel, self).__init__()
        self.char_model = char_model
        self.fc = nn.Linear(1280, 128)  # Combine features from both branches
        #self.fc = nn.Linear(1280, 128)  # When using a single branch
        self.classifier = nn.Linear(128, num_classes)  # Final classification layer

    def forward(self, char_input, context_input):
        char_features = self.char_model(char_input).view(char_input.size(0), -1)
        context_features = self.context_model(context_input).view(context_input.size(0), -1)
        combined_features = torch.cat((char_features, context_features), dim=1)
        combined_features = self.fc(combined_features)
        combined_features = F.relu(combined_features)  # Apply ReLU activation
        #char_features = self.fc(char_features) # code variation for single branch
        #char_features = F.relu(char_features) #code variation for single branch
        logits = self.classifier(combined_features)
        output = F.softmax(logits, dim=1)  # Apply softmax to get probabilities
        return output