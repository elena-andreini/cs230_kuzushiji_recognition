def create_pairs(features, labels):
    """
    Create pairs of samples for contrastive learning.

    :param features: List of features.
    :param labels: Corresponding labels for each feature.
    :return: Lists of pairs (anchor, positive/negative) and labels (1 for positive, 0 for negative).
    """
    pairs = []
    pair_labels = []

    num_samples = len(features)
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            if labels[i] == labels[j]:
                pairs.append((features[i], features[j]))
                pair_labels.append(1)  # Positive pair
            else:
                pairs.append((features[i], features[j]))
                pair_labels.append(0)  # Negative pair

    return pairs, pair_labels