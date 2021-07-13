# Input x Gradient interpretability method class. 
# Original papers: https://arxiv.org/pdf/1605.01713.pdf

import captum
import torch
import numpy as np

from interpretability_methods.interpretability_method import InterpretabilityMethod


class InputXGradient(InterpretabilityMethod):
    
    def __init__(self, model):
        super().__init__(model)
        self.ixg_method = captum.attr.InputXGradient(model)
        
    
    def get_masks(self, input_batch, target_classes=None):
        """Compute input x gradient mask."""
        if target_classes is None:
            target_classes = self.model(input_batch).argmax(dim=1)

        input_x_gradient = self.ixg_method.attribute(input_batch, 
                                                     target=target_classes)
        input_x_gradient = input_x_gradient.detach().cpu().numpy()
        return input_x_gradient