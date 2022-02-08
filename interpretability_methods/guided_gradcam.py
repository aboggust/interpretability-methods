"""
Guided GradCAM interpretability method class.
Original paper: https://arxiv.org/pdf/1610.02391.pdf
"""

import captum

from interpretability_methods.interpretability_method import InterpretabilityMethod


class GuidedGradCAM(InterpretabilityMethod):
    """Guided GradCAM interpretability method."""

    def __init__(self, model, layer):
        """
        Extends base method to include the saliency method.

        Additional Args:
        layer: the layer of the model to compute guided GradCAM with.
        """
        super().__init__(model)
        self.method = captum.attr.GuidedGradCam(model, layer)

    def get_saliency(self, input_batch, target_classes=None,
                     interpolation_method='bilinear'):
        """
        Extends base method to compute GradCAM.

        Additional Args:
        interpolation_method: method to upsample the GradCAM attributions. Can
            be "nearest", "area", "bilinear", or "bicubic". Defaults to
            "bilinear".
        """
        if target_classes is None:
            target_classes = self.model(input_batch).argmax(dim=1)

        guided_gradcam = self.method.attribute(
            input_batch,
            target=target_classes,
            interpolate_mode=interpolation_method
        )
        guided_gradcam = guided_gradcam.detach().cpu().numpy()
        return guided_gradcam
