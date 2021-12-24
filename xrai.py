# XRAI interpretability method class. From Google PAIR: https://github.com/PAIR-code/saliency
# Original paper: https://arxiv.org/abs/1906.02825

import saliency.core as saliency
import torch
import numpy as np

from interpretability_methods.interpretability_method import InterpretabilityMethod
from interpretability_methods.vanilla_gradients import VanillaGradients


class XRAI(InterpretabilityMethod):
    
    def __init__(self, model):
        super().__init__(model)
        self.method = saliency.XRAI()
        
    
    def get_masks(self, input_batch, target_classes=None):
        """Compute XRAI attributions."""        
        if target_classes is None:
            target_classes = self.model(input_batch).argmax(dim=1)
                        
        def call_model_function(x_value_batch, call_model_args=None, expected_keys=None):
            target_classes = call_model_args['target_classes']
            x_value_batch = x_value_batch.transpose(0, 3, 1, 2) # model takes channel first
            input_batch = torch.from_numpy(x_value_batch).to(self.device)
            gradients = VanillaGradients(self.model).get_masks(input_batch, target_classes, False)
            gradients = gradients.transpose(0, 2, 3, 1) # XRAI takes channel last
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}

        xrais = []
        for i, single_input in enumerate(input_batch):
            single_input = single_input.detach().cpu().numpy()
            single_input = single_input.transpose(1, 2, 0) # XRAI takes channel last
            xrai = self.method.GetMask(single_input, call_model_function, 
                                       {'target_classes': target_classes[i]})
            xrai = np.expand_dims(xrai, axis=0)
            xrais.append(xrai)
        return np.array(xrais)