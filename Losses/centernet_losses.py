import torch
from torch import nn

def focal_loss(pred, gt):
    alpha = 2
    beta = 4
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

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
	
	
def centernet_loss(pred_heatmap, pred_offset, pred_size, gt_heatmap, gt_offset, gt_size):
    #print(f'prediction shape {pred_heatmap.shape}')
    #print(f'gt heatmap shape {gt_heatmap.shape}')
    heatmap_loss = focal_loss(pred_heatmap, gt_heatmap)
    offset_loss = nn.functional.mse_loss(pred_offset, gt_offset, reduction='sum') / gt_heatmap.sum()
    size_loss = nn.functional.mse_loss(pred_size, gt_size, reduction='sum') / gt_heatmap.sum()
    return heatmap_loss + offset_loss +  size_loss