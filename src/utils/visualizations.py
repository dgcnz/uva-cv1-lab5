import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE


def plot_transormations(dataset, transform):
    # plots random images from the training set
    # with their transformations and without in 2x5 grid

    # 5 images from the training set
    dataset_cp = dataset
    dataset_cp.transform = None
    images = [dataset_cp[i][0] for i in range(5)]

    dataset_cp.transform = transform

    transformed_images = [dataset_cp[i][0] for i in range(5)]

    f, axarr = plt.subplots(2, 5)

    # changing size of the subplots
    f.set_figheight(4)
    f.set_figwidth(10)

    # adding title
    axarr[0, 2].set_title("Original images")
    axarr[1, 2].set_title("Transformed images")

    for i in range(5):
        axarr[0, i].imshow(images[i])
        axarr[1, i].imshow(transformed_images[i].permute(1, 2, 0))

    # removing the axis ticks
    for ax in axarr.flat:
        ax.set_xticks([])
        ax.set_yticks([])


def plot_cnn_activations(net, test_dataloader):
    """
    Plots the activations of all the convolutional layers of the network
    """
    # Ensure the model is in eval mode
    net.eval()

    net.to("cpu")

    # Get a random sample from the test set
    sample_input, _ = next(iter(test_dataloader))

    # Hook function to capture intermediate activations
    activations = []

    def hook_fn(module, input, output):
        activations.append(output)

    # Attach the hook to all convolutional layers
    hooks = []
    for layer in net.children():
        for sub_layer in layer.children():
            if isinstance(sub_layer, torch.nn.Conv2d):
                hooks.append(sub_layer.register_forward_hook(hook_fn))

    # Forward pass to get activations
    with torch.no_grad():
        _ = net(sample_input)

    # Detach the hooks after use
    for hook in hooks:
        hook.remove()

    plt.figure(figsize=(5, 4))

    sample_input = sample_input * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    plt.imshow(sample_input[0].permute(1, 2, 0).numpy())
    plt.title("Sample Image")
    plt.axis("off")
    plt.show()

    # Plot the activations
    for idx, activation in enumerate(activations):
        # Get the number of feature maps (channels)
        num_feature_maps = activation.size(1)

        # Plot each feature map
        for feature_map in range(num_feature_maps):
            plt.subplot(
                np.ceil(np.sqrt(num_feature_maps)).astype(int),
                np.ceil(np.sqrt(num_feature_maps)).astype(int),
                feature_map + 1,
            )
            plt.imshow(activation[0, feature_map].numpy(), cmap="viridis")
            plt.axis("off")

        plt.suptitle(f"Layer {idx+1} activations")
        plt.show()


def plot_embedded_space(_net, dataloader, class_names):
    """
    Plots the embedded space of the network using PCA.

    Args:
    - net (torch.nn.Module): The trained model.
    - dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
    - class_names (list): List of class names.
    """

    from copy import deepcopy

    net = deepcopy(_net)
    net._model = net._model[:-5]
    net._model

    net.to("cpu")

    # Ensure the network is in evaluation mode
    net.eval()

    # Lists to store features and labels
    features_list = []
    labels_list = []

    # Extract features for each batch
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs, _ = net(inputs)
            features_list.append(outputs.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    # Convert lists to numpy arrays
    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)

    # Apply TSNE to reduce dimensionality to 2D
    pca = TSNE(n_components=2)
    pca_result = pca.fit_transform(features)

    # Plot the 2D representation
    plt.figure(figsize=(10, 6))
    for i, class_name in enumerate(class_names):
        indices = np.where(labels == i)
        plt.scatter(pca_result[indices, 0], pca_result[indices, 1], label=class_name)
    plt.legend()
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
