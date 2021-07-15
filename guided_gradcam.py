# GradCAM interpretability method class. 
# Original paper: https://arxiv.org/pdf/1610.02391.pdf

import captum
import torch
import numpy as np

from interpretability_methods.interpretability_method import InterpretabilityMethod


class GuidedGradCAM(InterpretabilityMethod):
    
    def __init__(self, model, layer):
        super().__init__(model)
        self.gradcam_method = captum.attr.GuidedGradCam(model, layer)
        
    
    def get_masks(self, input_batch, target_classes=None, interpolate_model='bilinear'):
        """Compute Guided GradCAM attributions."""
        if target_classes is None:
            target_classes = self.model(input_batch).argmax(dim=1)

        guided_gradcam = self.gradcam_method.attribute(input_batch, target=target_classes, interpolate_mode=interpolate_model)
        guided_gradcam = guided_gradcam.detach().cpu().numpy()
        return guided_gradcam