# Vanilla gradients interpretability method class. 
# Derived from original papers: https://www.researchgate.net/profile/Aaron_Courville/publication/265022827_Visualizing_Higher-Layer_Features_of_a_Deep_Network/links/53ff82b00cf24c81027da530.pdf; https://arxiv.org/pdf/1312.6034.pdf

import torch
from interpretability_method import InterpretabilityMethod


class VanillaGradients(InterpretabilityMethod):
    
    def __init__(self, model):
        super().__init__(model)
    
    
    def get_mask(self, input_batch, smoothgrad):
        """Compute vanilla gradient mask."""
        input_batch.requires_grad_()
        output = self.model(input_batch)
        batch_scores, _ = output.max(dim=1)
        score_sum = torch.sum(batch_scores)
        score_sum.backward(retain_graph=True)
        vanilla_gradients = input_batch.grad.data
        assert(vanilla_gradients.shape == input_batch.shape)
        return vanilla_gradients.cpu().detach().numpy()