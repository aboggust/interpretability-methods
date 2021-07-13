# Guided Backprop interpretability method class. 
# Original paper: https://arxiv.org/pdf/1412.6806.pdf

import captum
import torch
import numpy as np

from interpretability_methods.interpretability_method import InterpretabilityMethod


class GuidedBackprop(InterpretabilityMethod):
    
    def __init__(self, model):
        super().__init__(model)
        self.gb_method = captum.attr.GuidedBackprop(model)
        
    
    def get_masks(self, input_batch, target_classes=None):
        """Compute guided backprop mask."""
        if target_classes is None:
            target_classes = self.model(input_batch).argmax(dim=1)

        guided_backprop = self.gb_method.attribute(input_batch, 
                                                   target=target_classes)
        guided_backprop = guided_backprop.detach().cpu().numpy()
        return guided_backprop