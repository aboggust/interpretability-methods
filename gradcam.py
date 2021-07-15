# GradCAM interpretability method class. 
# Original paper: https://arxiv.org/pdf/1610.02391.pdf

import captum
import torch
import numpy as np

from interpretability_methods.interpretability_method import InterpretabilityMethod


class GradCAM(InterpretabilityMethod):
    
    def __init__(self, model, layer):
        super().__init__(model)
        self.gradcam_method = captum.attr.LayerGradCam(model, layer)
        
    
    def get_masks(self, input_batch, target_classes=None):
        """Compute GradCAM attributions."""
        if target_classes is None:
            target_classes = self.model(input_batch).argmax(dim=1)

        gradcam = self.gradcam_method.attribute(input_batch, target=target_classes)
        upsampled_gradcam = captum.attr.LayerAttribution.interpolate(gradcam, 
                                                                     (input_batch.shape[-2], input_batch.shape[-1]),
                                                                     interpolate_mode='bilinear')
        upsampled_gradcam = upsampled_gradcam.detach().cpu().numpy()
        return upsampled_gradcam