"""
Gradient SHAP interpretability method class.
Original paper: https://arxiv.org/pdf/1705.07874.pdf
"""

import captum
import torch

from interpretability_methods.interpretability_method import InterpretabilityMethod


class GradientSHAP(InterpretabilityMethod):
    """Gradient SHAP interpretability method."""

    def __init__(self, model):
        """Extends base method to include the saliency method."""
        super().__init__(model)
        self.method = captum.attr.GradientShap(model)

    def get_saliency(self, input_batch, target_classes=None, baseline=None,
                     num_samples=5):
        """
        Extends base method to compute Gradient SHAP.

        Additional Args:
        baseline: None or torch Tensor of the same shape of the input_batch.
        num_samples: integer number of times to add white noise to the input.
        """
        if target_classes is None:
            target_classes = self.model(input_batch).argmax(dim=1)

        if baseline is None:
            baseline = torch.cat([input_batch * 0, input_batch * 1])
        baseline = baseline.to(self.device)

        shap = self.method.attribute(input_batch,
                                     baselines=baseline,
                                     target=target_classes,
                                     n_samples=num_samples)
        shap = shap.detach().cpu().numpy()
        return shap
