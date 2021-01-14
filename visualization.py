# Visualize interpretability outputs

import numpy as np
import matplotlib.pyplot as plt


def visualize_masks(batch, percentile=99):
    """
    Visualizes masks as 2D grayscale images.

    Args:
    batch: 4D numpy array (batch, channels, height, width) to visualize
    percentile: integer in range 0 to 100. Values above the percentile are clipped to 1.

    Returns:
    None and outputs a matplotlib figure with batch greyscale images showing the clipped masks in the batch.
    """
    masks = clip_masks(batch, percentile)
    fig = _plot_greyscale(masks)
    
    
def clip_masks(batch, percentile):
    batch_size = batch.shape[0]
    masks = np.sum(batch, axis=1)
    vmax = np.percentile(masks, percentile, axis=(1,2)).reshape((batch_size, 1, 1))
    vmin = np.min(masks, axis=(1,2)).reshape((batch_size, 1, 1))
    masks = np.clip((masks - vmin) / (vmax - vmin), 0, 1)
    return masks


def _plot_greyscale(masks):
    """Creates greyscale images from batch of 2D arrays."""
    num_masks, _, _ = masks.shape
    fig, ax = plt.subplots(ncols=num_masks)
    plt.axis('off')
    for i, mask in enumerate(masks):
        ax[i].set_axis_off()
        ax[i].imshow(mask, cmap=plt.cm.gray, vmin=0, vmax=1)
    return fig