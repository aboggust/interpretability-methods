"""Utility functions to run and visualize interpretability methods."""

import numpy as np
import matplotlib.pyplot as plt


def visualize_saliency(batch, percentile=99):
    """
    Visualizes saliency as 2D grayscale images.

    Args:
    batch: 4D numpy array (batch, channels, height, width) to visualize.
    percentile: integer in range 0 to 100. Values above the percentile are
        clipped to 1.

    Returns: None and outputs a matplotlib figure with batch greyscale images
        showing the clipped saliency in the batch.
    """
    saliency = clip_saliency(normalize_0to1(np.abs(batch)), percentile)
    fig = plot_greyscale(saliency)


def clip_saliency(batch, percentile):
    """
    Clip saliency at given percentile.

    Args:
    batch: 4D numpy array (batch, channels, height, width).
    percentile: integer in range 0 to 100. Values above the percentile are
        clipped to 1.

    Returns: A 4D numpy array scaled between 0 and 1 with all values above
    percentile set to 1.
    """
    batch_size = batch.shape[0]
    saliency = flatten_saliency(batch)
    vmax = np.percentile(saliency, percentile, axis=(1,2))
    vmax = vmax.reshape((batch_size, 1, 1))
    vmin = np.min(saliency, axis=(1,2))
    vmin = vmin.reshape((batch_size, 1, 1))
    saliency = np.clip((saliency - vmin) / (vmax - vmin), 0, 1)
    return saliency


def normalize_0to1(batch):
    """
    Normalize a batch such that every value is in the range 0 to 1.

    Args:
    batch: a batch first numpy array to be normalized.

    Returns: A numpy array of the same size as batch, where each item in the
    batch has 0 <= value <= 1.
    """
    axis = tuple(range(1, len(batch.shape)))
    minimum = np.min(batch, axis=axis).reshape((-1,) + (1,) * len(axis))
    maximum = np.max(batch, axis=axis).reshape((-1,) + (1,) * len(axis))
    normalized_batch = (batch - minimum) / (maximum - minimum)
    return normalized_batch


def plot_greyscale(batch):
    """
    Creates greyscale images from batch of 2D arrays.

    Args:
    batch: 4D numpy array (batch, channels, height, width) to visualize.

    Returns: A matplotlib figure displaying a grayscale image for each saliency
    in the batch.
    """
    batch_size, _, _ = batch.shape
    fig, ax = plt.subplots(ncols=batch_size)
    if batch_size == 1:
        ax = [ax]
    plt.axis('off')
    for i, mask in enumerate(batch):
        ax[i].set_axis_off()
        ax[i].imshow(mask, cmap=plt.cm.gray, vmin=0, vmax=1)
    return fig


def flatten_saliency(batch):
    """
    Flattens saliency by summing the channel dimension.

    Args:
    batch: 4D numpy array (batch, channels, height, width).

    Returns: 3D numpy array (batch, height, width) with the channel dimension
        summed.
    """
    return np.sum(batch, axis=1)
