# Vanilla gradients interpretability method class. 
# Derived from original papers: https://www.researchgate.net/profile/Aaron_Courville/publication/265022827_Visualizing_Higher-Layer_Features_of_a_Deep_Network/links/53ff82b00cf24c81027da530.pdf; https://arxiv.org/pdf/1312.6034.pdf

import torch
import numpy as np
from interpretability_method import InterpretabilityMethod


class VanillaGradients(InterpretabilityMethod):
    
    def __init__(self, model):
        super().__init__(model)
        
    
    def get_mask(self, input_instance, smoothgrad=False, target_class=None):
        """Compute vanilla gradient mask."""
        # Initialize gradient for the input
        input_instance.requires_grad_()
        if input_instance.grad is not None: # Zero out existing gradients
            input_instance.grad.data.zero_()

        # Compute the gradient of the input with respect to the target class
        output = self.model(input_instance)
        if not target_class:
            target_class = output.argmax(dim=1)
        output[torch.arange(output.shape[0]), target_class].backward(retain_graph=True)
        vanilla_gradients = input_instance.grad.data
        return vanilla_gradients.cpu().detach().numpy()
    