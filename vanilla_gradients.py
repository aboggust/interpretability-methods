# Vanilla gradients interpretability method class. 
# Original papers: https://www.researchgate.net/profile/Aaron_Courville/publication/265022827_Visualizing_Higher-Layer_Features_of_a_Deep_Network/links/53ff82b00cf24c81027da530.pdf; https://arxiv.org/pdf/1312.6034.pdf

import captum.attr
import torch
import numpy as np

from interpretability_methods.interpretability_method import InterpretabilityMethod


class VanillaGradients(InterpretabilityMethod):
    
    def __init__(self, model):
        super().__init__(model)
        self.vg_method = captum.attr.Saliency(model)
        
    
    def get_masks(self, input_batch, target_classes=None, absolute=True):
        """Compute vanilla gradient mask using the magnitude of the gradients."""
        if target_classes is None:
            target_classes = self.model(input_batch).argmax(dim=1)

        vanilla_gradients = self.vg_method.attribute(input_batch, 
                                                     target=target_classes, 
                                                     abs=absolute)
        vanilla_gradients = vanilla_gradients.detach().cpu().numpy()
        return vanilla_gradients