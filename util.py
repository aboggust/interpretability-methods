# Utility functions for the interpretability methods.

import numpy as np


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
    return (batch - minimum) / (maximum - minimum)