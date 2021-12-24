# RISE interpretability method class. From: https://github.com/eclique/RISE
# Original paper: https://arxiv.org/abs/1806.07421

import torch
import numpy as np

from interpretability_methods.interpretability_method import InterpretabilityMethod
from interpretability_methods.rise_master import explanations


class RISE(InterpretabilityMethod):
    
    def __init__(self, model, input_size, num_masks=500):
        super().__init__(model)
        self.method = explanations.RISE(model, input_size, 56) # GPU batch is set as in example
        
        maskspath = '/home/aboggust/data/temp/masks.npy'
        self.method.generate_masks(N=num_masks, s=8, p1=0.1, savepath=maskspath)

    
    def get_masks(self, input_batch, target_classes=None):
        """Compute RISE attributions."""        
        if target_classes is None:
            target_classes = self.model(input_batch).argmax(dim=1).cpu().numpy()
        
        rise_masks = []
        for i, single_input in enumerate(input_batch):
            rise_mask = self.method(single_input.unsqueeze(0)).cpu().numpy()
            rise_mask = rise_mask[target_classes[i]]
            if len(target_classes) == 1:
                rise_mask = np.expand_dims(rise_mask, axis=0)            
            rise_masks.append(rise_mask)
            
        rise_masks = np.expand_dims(np.array(rise_masks), axis=1)
        return rise_masks