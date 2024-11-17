import matplotlib.pyplot as plt

def show_images(images, labels, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))

    for i in range(num_images):
        # Display color images
        ax = axes[i]
        ax.imshow(images[i].permute(1, 2, 0))  # No need for `squeeze()` since we're working with color images
        ax.set_title(labels[i])
        ax.axis('off')

    plt.show()

