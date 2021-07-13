# Integrated gradients interpretability method class. 
# Original paper: https://arxiv.org/pdf/1703.01365.pdf

import captum
import numpy as np
import torch

from interpretability_methods.interpretability_method import InterpretabilityMethod

class IntegratedGradients(InterpretabilityMethod):
    
    def __init__(self, model):
        super().__init__(model)
        self.ig_method = captum.attr.IntegratedGradients(self.model)

        
    def get_masks(self, input_batch, target_classes=None, baseline=None, num_points=50):
        """
        Compute integrated gradient mask.
        
        Additional Args:
        baseline: None or torch Tensor of the same shape of the input_batch representing
            the baseline input to integrate from.
        num_points: integer number of points to integrate from the baseline to the input.
        """
        if baseline is None:
            baseline = torch.randn(input_batch.shape)
        baseline = baseline.to(self.device)
        
        if target_classes is None: # compute each point with respect to the input's predicted class
            target_classes = self.model(input_batch).argmax(dim=1)
            
        integrated_gradients = self.ig_method.attribute(input_batch, 
                                                        baselines=baseline, 
                                                        target=target_classes,
                                                        n_steps=num_points)
        integrated_gradients = integrated_gradients.detach().cpu().numpy()
        return integrated_gradients
