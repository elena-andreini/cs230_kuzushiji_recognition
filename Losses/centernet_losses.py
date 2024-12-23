import torch
from torch import nn
from DataSets.datasets import draw_gaussian

def focal_loss(pred, gt, pos_threshold=0.5):
    alpha = 2
    beta = 4
    pos_inds = gt.ge(pos_threshold)
    neg_inds = gt.lt(pos_threshold)

    neg_weights = torch.pow(1 - gt[neg_inds], beta)
    loss = 0

    pos_loss = torch.log(pred[pos_inds]) * torch.pow(1 - pred[pos_inds], alpha)
    neg_loss = torch.log(1 - pred[neg_inds]) * torch.pow(pred[neg_inds], alpha) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss
	
	

def adaptive_size_loss(pred_size, gt_size):
    error = torch.abs(pred_size - gt_size)
    penalty = torch.exp(error) - 1  # Exponential penalty
    return (penalty * error).mean()



def log_cosh_loss(pred, target):
    loss = torch.log(torch.cosh(pred - target))
    return loss.mean()

def huber_loss(pred, target, delta=1.0): 
    return nn.functional.smooth_l1_loss(pred, target, reduction='mean', beta=delta)
  

def centernet_loss(pred_heatmap, pred_offset, pred_size, gt_heatmap, gt_offset, gt_size, pos_threshold=1.0):
    # Count the number of keypoints 
    num_pos = gt_heatmap.gt(pos_threshold).float().sum()
    pos_mask = gt_heatmap.gt(pos_threshold).expand(-1, 2, -1, -1)
    heatmap_loss = focal_loss(pred_heatmap, gt_heatmap, pos_threshold=pos_threshold)
    # Calculate offset loss and size loss 
    if num_pos > 0: 
        offset_loss = nn.functional.mse_loss(pred_offset[pos_mask], gt_offset[pos_mask], reduction='sum') / num_pos
        size_loss = nn.functional.mse_loss(pred_size[pos_mask], gt_size[pos_mask], reduction='sum') / num_pos        
    else: 
        offset_loss = torch.tensor(0.0, device=pred_heatmap.device) 
        size_loss = torch.tensor(0.0, device=pred_heatmap.device) 
       
    #offset_loss = nn.functional.mse_loss(pred_offset, gt_offset, reduction='sum') / gt_heatmap.sum()
    #size_loss = nn.functional.mse_loss(pred_size, gt_size, reduction='sum') / gt_heatmap.sum()
    #size_loss = adaptive_size_loss(pred_size, gt_size) / gt_heatmap.sum()
    #size_loss = huber_loss(pred_size, gt_size)
    
    return heatmap_loss + offset_loss +  size_loss