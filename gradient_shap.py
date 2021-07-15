# Gradient SHAP interpretability method class. 
# Original paper: https://arxiv.org/pdf/1705.07874.pdf
# Kernel SHAP.

import captum
import torch
import numpy as np

from interpretability_methods.interpretability_method import InterpretabilityMethod


class GradientSHAP(InterpretabilityMethod):
    
    def __init__(self, model):
        super().__init__(model)
        self.shap_method = captum.attr.GradientShap(model)
        
    
    def get_masks(self, input_batch, target_classes=None, baseline=None, n_samples=5):
        """Compute Gradient SHAP mask."""
        if target_classes is None:
            target_classes = self.model(input_batch).argmax(dim=1)
            
        if baseline is None:
            baseline = torch.cat([input_batch * 0, input_batch * 1])
        baseline = baseline.to(self.device)

        shap = self.shap_method.attribute(input_batch,
                                          baselines=baseline,
                                          target=target_classes,
                                          n_samples=n_samples,)
        shap = shap.detach().cpu().numpy()
        return shap