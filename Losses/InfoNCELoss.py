import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

class InfoNCELoss(nn.Module):
    def __init__(self, temperature, device):
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, query, pos, neg):
        '''
        Expects query and pos of size (BS, 1, E) and neg of 
		size (BS, N, E) where BS is the batch size, N is the 
		number of negative samples, E is the dimension of the embedded
		size and can be either an integer or a series of numbers

        '''
		
		BS = query.shape[0]
		N = neg.shape[1]
		
		logit_pos = torch.nn.functional.cosine_similarity(query.view(BS, -1), pos.view(BS, -1))
		logit_neg = torch.nn.functinoal.cosine_similarity(query.view(BS, -1), neg.view(BS, N, -1))
		
        # Concatenate logits
        logits = torch.cat((logits_pos, logits_neg), dim=1)
        
        # Generate labels
       
        labels = torch.zeros(BS).to(self.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits / self.temperature, labels, reduction='mean')

        return loss