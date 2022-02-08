"""
Guided Backprop interpretability method class.
Original paper: https://arxiv.org/pdf/1412.6806.pdf
"""

import captum

from interpretability_methods.interpretability_method import InterpretabilityMethod


class GuidedBackprop(InterpretabilityMethod):
    """Guided backprop interpretability method."""

    def __init__(self, model):
        """Extends base method to include the saliency method."""
        super().__init__(model)
        self.method = captum.attr.GuidedBackprop(model)

    def get_saliency(self, input_batch, target_classes=None):
        """Extends base method to compute guided backprop saliency."""
        if target_classes is None:
            target_classes = self.model(input_batch).argmax(dim=1)

        guided_backprop = self.method.attribute(input_batch,
                                                target=target_classes)
        guided_backprop = guided_backprop.detach().cpu().numpy()
        return guided_backprop
    