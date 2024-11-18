import torch

def custom_collate_fn(batch):
    images, gt_heatmaps, gt_offsets, gt_sizes, boxes = zip(*batch)

    # Convert images, heatmaps, offsets, sizes to tensors
    images = torch.stack(images)
    gt_heatmaps = torch.stack(gt_heatmaps)
    gt_offsets = torch.stack(gt_offsets)
    gt_sizes = torch.stack(gt_sizes)

    # No need to stack boxes, keep them as a list of lists
    return images, gt_heatmaps, gt_offsets, gt_sizes, boxes