# Parent class for all interpretability methods

from abc import ABC, abstractmethod


class InterpretabilityMethod(ABC):
    """Base class for all interpretability methods."""
    
    def __init__(self, model):
        """ TODO """
        self.model = model
        self.model.eval()
    
    
    @abstractmethod
    def get_mask(self, input_instance, smoothgrad=False, target_class=None):
        """
        Returns mask of activations for the given input.
        
        Args:
        input_instance: inputs as torch array to run interpretability method on.
        smoothgrad: boolean value indicating if smoothgrad should be applied.
        target_class: None or integer indicating the target class for the input.
            If None, then the predicted class is used as the target class.
        
        Returns:
        A numpy array the same shape as input_instance of interpretability activations with respect
        to the target class. Output is not normalized or in absolute values.
        """
        pass
    
    
    def _apply_smoothgrad(self, mask):
        """
        TODO: Applies smoothgrad to mask.
        """
        pass