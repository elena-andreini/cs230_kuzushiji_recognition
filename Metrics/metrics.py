import torch

def cosine_similarity(vec1, vec2):
    vec1 = torch.tensor(vec1, dtype=torch.float32)
    vec2 = torch.tensor(vec2, dtype=torch.float32)
    dot_product = torch.dot(vec1, vec2)
    # Calculate the L2 norms
    norm_vec1 = torch.norm(vec1)
    norm_vec2 = torch.norm(vec2)
    cosine_sim = dot_product / (norm_vec1 * norm_vec2)
    return cosine_sim.item()

def euclidean_distance(vec1, vec2):
    vec1 = torch.tensor(vec1, dtype=torch.float32)
    vec2 = torch.tensor(vec2, dtype=torch.float32)
    dot_product = torch.dot(vec1, vec2)
    dist = np.linalg.norm(vec1 - vec2)
    return dist
    
    


def calculate_accuracy(output, labels):
    """
    For models with terminal softmax layer
    """
    _, preds = torch.max(output, dim=1)
    correct = torch.sum(preds == labels)
    return correct.item() / len(labels)