# Visualize interpretability outputs

import numpy as np
import matplotlib.pyplot as plt


def visualize_masks(batch, percentile=99):
    """
    Visualizes masks as 2D grayscale images.

    Args:
    batch: 4D numpy array (batch, channels, height, width) to visualize.
    percentile: integer in range 0 to 100. Values above the percentile are clipped to 1.

    Returns:
    None and outputs a matplotlib figure with batch greyscale images showing the clipped masks in the batch.
    """
    masks = clip_masks(normalize_0to1(np.abs(batch)), percentile)
    fig = plot_greyscale(masks)
    

def flatten_masks(batch):
    """
    Flattens masks by summing the channel dimension.

    Args:
    batch: 4D numpy array (batch, channels, height, width).

    Returns:
    3D numpy array (batch, height, width) with the channel dimension summed.
    """
    return np.sum(batch, axis=1)


def binarize_masks_percentile(batch, percentile):
    """
    Creates binary mask by thresholding at threshold.

    Args:
    batch: 4D numpy array (batch, channels, height, width).
    threshold: float in range 0 to 1. Values above the threshold are set to 1.
        Values below the threshold are set to 0.

    Returns:
    A 4D numpy array with dtype uint8 with all values set to 0 or 1.
    """
    batch_size = batch.shape[0]
    batch_normalized = normalize_0to1(batch)
    percentile = np.percentile(batch_normalized, threshold * 100, axis=(1, 2, 3)).reshape(batch_size, 1, 1, 1)
    binary_mask = (batch_normalized >= percentile).astype('uint8')
    return binary_mask


def binarize_masks(batch, num_std=1):
    """
    Creates binary mask by thresholding at one standard deviation above the mean.

    Args:
    batch: 3D numpy array (batch, height, width).
    threshold: float in range 0 to 1. Values above the threshold are set to 1.
        Values below the threshold are set to 0.

    Returns:
    A 3D numpy array with dtype uint8 with all values set to 0 or 1.
    """
    batch_size = batch.shape[0]
    batch_normalized = normalize_0to1(batch)
    mean = np.mean(batch_normalized, axis=(1, 2)).reshape(batch_size, 1, 1)
    std = np.std(batch_normalized, axis=(1, 2)).reshape(batch_size, 1, 1)
    threshold = mean + num_std * std
    binary_mask = (batch_normalized >= threshold).astype('uint8')
    return binary_mask

    
def clip_masks(batch, percentile):
    """
    Clip masks at given percentile.

    Args:
    batch: 4D numpy array (batch, channels, height, width).
    percentile: integer in range 0 to 100. Values above the percentile are clipped to 1.

    Returns:
    A 4D numpy array scaled between 0 and 1 with all values above percentile set to 1.
    """
    batch_size = batch.shape[0]
    masks = flatten_masks(batch)
    vmax = np.percentile(masks, percentile, axis=(1,2)).reshape((batch_size, 1, 1))
    vmin = np.min(masks, axis=(1,2)).reshape((batch_size, 1, 1))
    masks = np.clip((masks - vmin) / (vmax - vmin), 0, 1)
    return masks


def plot_greyscale(batch):
    """
    Creates greyscale images from batch of 2D arrays.

    Args:
    batch: 4D numpy array (batch, channels, height, width) to visualize.

    Returns:
    A matplotlib figure displaying a grayscale image for each mask in the plot.
    """
    """"""
    num_masks, _, _ = batch.shape
    fig, ax = plt.subplots(ncols=num_masks)
    if num_masks == 1:
        ax = [ax]
    plt.axis('off')
    for i, mask in enumerate(batch):
        ax[i].set_axis_off()
        ax[i].imshow(mask, cmap=plt.cm.gray, vmin=0, vmax=1)
    return fig


def normalize_0to1(batch):
    """
    Normalize a batch of matrices such that every value is in the range 0 to 1.
    
    Args:
        batch: a batch first numpy array to be normalized.
        
    Returns:
        A numpy array of the same size as batch, where each matrix in the batch has
        0 <= value <= 1.
    """
    axis = tuple(range(1, len(batch.shape)))
    minimum = np.min(batch, axis=axis).reshape((-1,) + (1,) * len(axis))
    maximum = np.max(batch, axis=axis).reshape((-1,) + (1,) * len(axis))
    normalized_batch = (batch - minimum) / (maximum - minimum)
    return normalized_batch