# Visualize interpretability outputs

import numpy as np
import matplotlib.pyplot as plt


def visualize_mask(mask, percentile=99, plot=True):
    """
    Visualizes mask as a 2D grayscale image.

    Args:
    mask: 3D numpy array (channels, height, width) to visualize
    percentile: integer in range 0 to 100. Values above the percentile are clipped to 1.
    plot: boolean representing whether the output mask is plotted.

    Returns:
    A 2D numpy array (heigh, width) by mazimizing over the absolute values of the channel 
    dimension. Outputs a greyscale image of the output if plot is True.
    """
    mask_2d = np.sum(np.abs(mask), axis=0)
    vmax = np.percentile(mask_2d, percentile)
    vmin = np.min(mask_2d)
    mask_2d = np.clip((mask_2d - vmin) / (vmax - vmin), 0, 1)
    if plot:
        plt.figure()
        plt.axis('off')
        plt.imshow(mask_2d, cmap=plt.cm.gray, vmin=0, vmax=1)
    return mask_2d