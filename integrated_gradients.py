# Integrated gradients interpretability method class. 
# Derived from original paper: https://arxiv.org/pdf/1703.01365.pdf

import numpy as np
import torch

from interpretability_method import InterpretabilityMethod
from vanilla_gradients import VanillaGradients


class IntegratedGradients(InterpretabilityMethod):
    
    def __init__(self, model):
        super().__init__(model)
    
    def get_masks(self, input_batch, target_classes=None, baseline=None, num_points=25):
        """
        Compute integrated gradient mask.
        
        Additional Args:
        baseline: None or torch Tensor of the same shape of the input_batch representing
            the baseline input to integrate from.
        num_points: integer number of points to integrate from the baseline to the input.
        """
        if baseline is None:
            baseline = torch.zeros(input_batch.shape).to(self.device)
        if target_classes is None: # compute each point with respect to the input's predicted class
            target_classes = self.model(input_batch).argmax(dim=1)
        vanilla_gradients = VanillaGradients(self.model)
        
        cumulative_gradients = np.zeros(input_batch.shape)
        for alpha in torch.linspace(0, 1, num_points):
            input_point = baseline + alpha * (input_batch - baseline)
            point_gradient = vanilla_gradients.get_masks(input_point, 
                                                         target_classes=target_classes)
            cumulative_gradients += point_gradient
        integrated_gradients = cumulative_gradients * (input_batch.detach().cpu().numpy() - baseline.detach().cpu().numpy()) / num_points
        return integrated_gradients
