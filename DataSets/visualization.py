import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_image(image_path):
    # Read the image from the given path
    img = mpimg.imread(image_path)
    
    # Display the image
    plt.imshow(img)
    plt.axis('off')  # Turn off axis
    plt.show()



def show_images(images, labels, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))

    for i in range(num_images):
        # Display color images
        ax = axes[i]
        ax.imshow(images[i].permute(1, 2, 0))  # No need for `squeeze()` since we're working with color images
        ax.set_title(labels[i])
        ax.axis('off')

    plt.show()



def show_images_1(imgs, labels):
    print(f'show_images_1 - imgs.shape {imgs.shape}')
    if imgs.ndim == 3: # Add a new dimension to handle the single image case
        imgs = imgs.unsqueeze(0)
        labels = [labels]
    fig, axes = plt.subplots(1, imgs.shape[0], figsize=(15, 15))
    # If there's only one image, axes will not be an array
    if imgs.shape[0] == 1:
        axes = [axes]
    imgs_list = [imgs[i] for i in range(imgs.shape[0])]
    for img, label, ax in zip(imgs_list, labels, axes):
        ax.imshow(img.permute(1, 2, 0).squeeze(), cmap='gray')
        # Add label to the image
        ax.text(10, 10, label, color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        ax.axis('off')
