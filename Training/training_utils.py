import torch


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
    
    

def calculate_accuracy(output, labels):
    _, preds = torch.max(output, dim=1)
    correct = torch.sum(preds == labels)
    return correct.item() / len(labels)