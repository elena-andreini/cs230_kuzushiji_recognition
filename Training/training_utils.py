import torch
import random

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
    
    

def calculate_accuracy(output, labels):
    _, preds = torch.max(output, dim=1)
    correct = torch.sum(preds == labels)
    return correct.item() / len(labels)
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def freeze_model(model):
    """
    Freezes all parameters in model.
    Parameters:
    model (torch.nn.Module): The model to be frozen.
    """
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_last_n_modules(model, n):
    """
    Unfreezes the last n modules in the given model.

    Parameters:
    model (torch.nn.Module): The model to unfreeze.
    n (int): Number of last modules to unfreeze.
    """
    # Get all modules in the model
    modules = list(model.children())

    # Check if n is greater than the number of modules
    if n > len(modules):
        print(f"Warning: The model only has {len(modules)} modules.")
        n = len(modules)
    #print(f'the model has {modules} modules')
    # Unfreeze the last n modules
    for module in modules[-n:]:
        for param in module.parameters():
            param.requires_grad = True


