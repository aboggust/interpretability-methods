"""
LIME interpretability method class.
Original paper: https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf
"""

import numpy as np
import torch
import torch.nn.functional as F
from lime import lime_image

from interpretability_methods.interpretability_method import InterpretabilityMethod


class LIME(InterpretabilityMethod):
    """LIME interpretability method."""

    def __init__(self, model, to_pil_transform, to_tensor_transform):
        """
        Extends base method to include the saliency method.

        Additional Args:
        to_pil_transform: torch transform that takes in a tensor consumable by
            the model and outputs the original unnormalized PIL image.
        to_tensor_transform: torch transform that takes in an unnormalized image
            and outputs a tensor consumable by the model (likely normalized).
        """
        super().__init__(model)
        self.method = lime_image.LimeImageExplainer()
        self.to_pil_transform = to_pil_transform
        self.to_tensor_transform = to_tensor_transform
        self.num_channels = None # updated when input batch is run

    def get_saliency(self, input_batch, target_classes=None, num_samples=1000,
                     positive_only=True):
        """
        Extends base method to compute LIME.

        Additional Args:
        num_samples: integer number of samples to use to train the surrogate
            model with. Defaults to 1000.
        positive_only: a boolean that if True, only returns attribution in
            support of the target class. If False, returns all atrribution.
            Defaults to True.
        """
        batch_size, num_channels, height, width = input_batch.shape
        self.num_channels = num_channels
        masks = np.empty((batch_size, 1, height, width))
        top_labels = 1
        if target_classes is not None:
            top_labels = None

        input_batch = input_batch.detach().cpu()
        for i, instance in enumerate(input_batch):
            labels = None
            if target_classes is not None:
                labels = [target_classes[i]]
            explanation = self.method.explain_instance(
                np.array(self.to_pil_transform(instance)),
                self._batch_predict,
                labels=labels,
                top_labels=top_labels,
                hide_color=0,
                num_samples=num_samples,
            )
            if labels is None:
                label = explanation.top_labels[0]
            else:
                label = labels[0]
            masks[i] = self._get_map(explanation, label,
                                     positive_only=positive_only)
        return masks

    def _batch_predict(self, input_batch):
        """Batch predict function required by LIME."""
        self.model = self.model.to(self.device)
        input_batch = torch.stack(
            tuple(self.to_tensor_transform(i) for i in input_batch),
            dim=0
        )
        if self.num_channels == 1:
            input_batch = input_batch[:, 0:1, :, :]
        input_batch = input_batch.to(self.device)
        output = self.model(input_batch)
        probabilities = F.softmax(output, dim=1)
        return probabilities.detach().cpu().numpy()

    def _get_map(self, explanation, label, positive_only):
        """Use LIME explanation internals to get feature-level attribution."""
        segments = explanation.segments
        mask = np.zeros(segments.shape)
        feature_explanations = explanation.local_exp[label]
        for feature, saliency in feature_explanations:
            if positive_only:
                saliency = max(saliency, 0)
            mask[segments == feature] = saliency
        return np.expand_dims(mask, axis=0)
