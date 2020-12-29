# Parent class for all interpretability methods

from abc import ABC, abstractmethod
import numpy as np
import torch


class InterpretabilityMethod(ABC):
    """Base class for all interpretability methods."""
    
    def __init__(self, model):
        """ TODO """
        self.model = model
        self.model.eval()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    @abstractmethod
    def get_mask(self, input_instance, target_class=None):
        """
        Returns mask of activations for the given input.
        
        Args:
        input_instance: inputs as torch array to run interpretability method on.
        target_class: None or integer indicating the target class for the input.
            If None, then the predicted class is used as the target class.
        
        Returns:
        A numpy array the same shape as input_instance of interpretability activations with respect
        to the target class. Output is not normalized or in absolute values.
        """
        pass
    
    
    def get_smoothgrad_mask(self, input_instance, target_class=None, std_spread=.15, num_samples=25, magnitude=True):
        """
        Returns a mask of smoothgrad activations for the given input.
        
        Args:
        input_instance: inputs as torch array to run interpretability method on.
        target_class: None or integer indicating the target class for the input.
            If None, then the predicted class is used as the target class.
        std_spread: amount of noise to add to the input, as fraction of the total spread.
        num_samples: number of samples to average across to get the smooth gradient.
        magnitude: boolean representing whether to compute sum of squares.
        
        Returns:
        A numpy array the same shape as input_instance of smoothgrad interpretability activations with respect
        to the target class. Output is not normalized or in absolute values.
        
        """
        std = float(std_spread * (torch.max(input_instance) - torch.min(input_instance)))
        total_gradients = np.zeros(input_instance.shape)
        for i in range(num_samples):
            noise = torch.empty(input_instance.shape).normal_(mean=0, std=std).to(self.device)
            noisy_input_instance = input_instance + noise
            noisy_input_instance.requires_grad_()
            noisy_input_instance.retain_grad()
            gradients = self.get_mask(noisy_input_instance, target_class=target_class)
            if magnitude:
                total_gradients += (gradients * gradients)
            else:
                total_gradients += gradients
        return total_gradients / num_samples