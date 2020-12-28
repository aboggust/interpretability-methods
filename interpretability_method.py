# Parent class for all interpretability methods

from abc import ABC, abstractmethod


class InterpretabilityMethod(ABC):
    """Base class for all interpretability methods."""
    
    def __init__(self, model):
        """ TODO """
        self.model = model
    
    
    @abstractmethod
    def get_mask(self, input_batch, smoothgrad):
        """
        Returns mask of activations for the given input.
        
        Args:
        input_batch: batched inputs as torch array to run interpretability method on.
        smoothgrad: boolean value indicating if smoothgrad should be applied.
        
        Returns:
        A numpy array the same shape as input_batch of interpretability activations.
        Output is not normalized or in absolute values.
        """
        pass
    
    
    def _apply_smoothgrad(self, mask):
        """
        TODO: Applies smoothgrad to mask.
        """
        pass