# LIME interpretability method class. 
# Derived from original paper: https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf
from lime import lime_image
import numpy as np
import torch
import torch.nn.functional as F

from interpretability_methods.interpretability_method import InterpretabilityMethod


class LIME(InterpretabilityMethod):
    
    def __init__(self, model, to_pil_transform, to_tensor_transform):
        super().__init__(model)
        self.explainer = lime_image.LimeImageExplainer()
        self.to_pil_transform = to_pil_transform
        self.to_tensor_transform = to_tensor_transform
    
    def get_masks(self, input_batch, target_classes=None, threshold=None, num_samples=1000, positive_only=True):
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
            explanation = self.explainer.explain_instance(np.array(self.to_pil_transform(instance)),
                                                          self._batch_predict,
                                                          labels=labels,
                                                          top_labels=top_labels,
                                                          hide_color=0,
                                                          num_samples=num_samples,
                                                          num_features=np.prod(instance.shape))
            if labels is None:
                label = explanation.top_labels[0]
            else:
                label = labels[0]
            masks[i] = self._get_map(explanation, label, positive_only=positive_only)
        return masks
                       
        
    def _batch_predict(self, input_batch):
        """Batch predict function required by LIME."""
        self.model = self.model.to(self.device)
        input_batch = torch.stack(tuple(self.to_tensor_transform(i) for i in input_batch), dim=0)
        if self.num_channels == 1:
            input_batch = input_batch[:, 0:1, :, :]
        input_batch = input_batch.to(self.device)
        output = self.model(input_batch)
        probabilities = F.softmax(output, dim=1)
        return probabilities.detach().cpu().numpy()
    
    
    def _get_map(self, explanation, label, positive_only=True):
        """Use LIME explanation internals to get saliency map as opposed to mask."""
        segments = explanation.segments
        mask = np.zeros(segments.shape)
        feature_explanations = explanation.local_exp[label]
        for feature, saliency in feature_explanations:
            if positive_only:
                saliency = max(saliency, 0)
            mask[segments == feature] = saliency
        return np.expand_dims(mask, axis=0)
        