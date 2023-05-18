"""Tests for interpretability methods."""

import os
import unittest
from unittest.mock import patch
import numpy as np
import PIL.Image
import torch
from torchvision import models

from interpretability_methods.vanilla_gradients import VanillaGradients
from interpretability_methods.integrated_gradients import IntegratedGradients


class TestInterpretabilityMethod(unittest.TestCase):
    """Test the base interpretability_method class."""

    def setUp(self):
        """Set up tests using Inception V3 and VanillaGradients instance."""
        model = models.__dict__['inception_v3'](pretrained=True).cuda().eval()
        test_image = os.path.join(os.path.dirname(__file__), 'doberman.png')
        image_array = np.asarray(PIL.Image.open(test_image)) / 127.5 - 1.0
        self.image = image_array.transpose(2, 0, 1)
        self.vanilla_gradients = VanillaGradients(model)
        self.tolerace = 1e-05

    def test_get_saliency_smoothed_single_input(self):
        """Test smoothgrad function output."""
        input_batch = torch.from_numpy(self.image).cuda().float().unsqueeze(0)

        # Test output of smoothgrad function.
        smoothgrad_gradients = self.vanilla_gradients.get_saliency_smoothed(
            input_batch
        )
        self.assertEqual(smoothgrad_gradients.shape, input_batch.shape)
        self.assertIs(type(smoothgrad_gradients), np.ndarray)

        # Test smoothgrad gradients are unequal to vanilla gradients.
        gradients = self.vanilla_gradients.get_saliency(input_batch)
        self.assertFalse(np.allclose(gradients,
                                     smoothgrad_gradients,
                                     atol=self.tolerace))

        # Test one run of smoothgrad without noise equals to gradients squared.
        with patch('torch.Tensor.normal_') as mocked_noise:
            mocked_noise.return_value = torch.zeros(input_batch.shape)
            equivalent_gradients = self.vanilla_gradients.get_saliency_smoothed(
                input_batch, num_samples=1, magnitude=False
            )
            self.assertTrue(np.allclose(gradients,
                                        equivalent_gradients,
                                        atol=self.tolerace))

    def test_get_saliency_smoothed_batched_input(self):
        """Test smoothgrad function output on batched inputs."""
        image = torch.from_numpy(self.image).cuda().float()
        input_duplicate = torch.from_numpy(self.image).cuda().float()
        image_flip = torch.from_numpy(
            np.flip(self.image, axis=1).copy()
        ).cuda().float()
        input_batch = torch.stack([image, input_duplicate, image_flip])

        # Test output of smoothgrad function.
        smoothgrad = self.vanilla_gradients.get_saliency_smoothed(
            input_batch
        )
        self.assertEqual(smoothgrad.shape, input_batch.shape)
        self.assertIs(type(smoothgrad), np.ndarray)

        # Test first two gradients are equal and unequal to the flipped input.
        with patch('torch.Tensor.normal_') as mocked_noise:
            mocked_noise.return_value = torch.zeros(input_batch.shape)
            smoothgrad_no_noise = self.vanilla_gradients.get_saliency_smoothed(
                input_batch, magnitude=False
            )
            self.assertTrue(np.allclose(smoothgrad_no_noise[0],
                                        smoothgrad_no_noise[1],
                                        atol=self.tolerace))
            self.assertFalse(np.allclose(smoothgrad_no_noise[0],
                                         smoothgrad_no_noise[2],
                                         atol=self.tolerace))
            self.assertFalse(np.allclose(smoothgrad_no_noise[1],
                                         smoothgrad_no_noise[2],
                                         atol=self.tolerace))


class TestVanillaGradients(unittest.TestCase):
    """Test the Vanilla Gradients class."""

    def setUp(self):
        model = models.__dict__['inception_v3'](pretrained=True).cuda().eval()
        test_image = os.path.join(os.path.dirname(__file__), 'doberman.png')
        image_array = np.asarray(PIL.Image.open(test_image)) / 127.5 - 1.0
        image = image_array.transpose(2, 0, 1)
        image_flip = np.flip(image, axis=1).copy()
        input_image = torch.from_numpy(image).cuda().float()
        input_image_flip = torch.from_numpy(image_flip).cuda().float()
        self.input_batch = torch.stack([input_image, input_image_flip])
        self.vanilla_gradients = VanillaGradients(model)
        self.tolerace = 1e-05

    def test_get_saliency(self):
        """Tests output shape and type are correct for single input."""
        gradients = self.vanilla_gradients.get_saliency(self.input_batch)
        self.assertEqual(gradients.shape, self.input_batch.shape)
        self.assertIs(type(gradients), np.ndarray)

    def test_get_saliency_target_classes(self):
        """Test default and given values of target classes."""
        gradients_predicted = self.vanilla_gradients.get_saliency(
            self.input_batch, target_classes=None
        )
        self.assertEqual(gradients_predicted.shape, self.input_batch.shape)
        self.assertIs(type(gradients_predicted), np.ndarray)

        gradients_class_100 = self.vanilla_gradients.get_saliency(
            self.input_batch, target_classes=[100, 100]
        )
        self.assertEqual(gradients_class_100.shape, self.input_batch.shape)
        self.assertIs(type(gradients_class_100), np.ndarray)

        gradients_class_236 = self.vanilla_gradients.get_saliency(
            self.input_batch, target_classes=[236, 236]
        )
        self.assertEqual(gradients_class_236.shape, self.input_batch.shape)
        self.assertIs(type(gradients_class_236), np.ndarray)

        # Input images predicted classes are 236.
        self.assertFalse(np.allclose(gradients_predicted,
                                     gradients_class_100,
                                     atol=self.tolerace))
        self.assertTrue(np.allclose(gradients_predicted,
                                    gradients_class_236,
                                    atol=self.tolerace))
        self.assertFalse(np.allclose(gradients_class_100,
                                     gradients_class_236,
                                     atol=self.tolerace))

    def test_get_saliency_consistency(self):
        """Tests multiple runs return the same gradients."""
        gradients_one = self.vanilla_gradients.get_saliency(self.input_batch)
        self.assertEqual(gradients_one.shape, self.input_batch.shape)
        self.assertIs(type(gradients_one), np.ndarray)

        gradients_two = self.vanilla_gradients.get_saliency(self.input_batch)
        self.assertEqual(gradients_two.shape, self.input_batch.shape)
        self.assertIs(type(gradients_two), np.ndarray)

        self.assertTrue(np.allclose(gradients_one,
                                    gradients_two,
                                    atol=self.tolerace))


class TestIntegratedGradients(unittest.TestCase):
    """Test the Integrated Gradients class."""

    def setUp(self):
        model_name = 'inception_v3'
        self.model = models.__dict__[model_name](pretrained=True).cuda().eval()
        test_image = os.path.join(os.path.dirname(__file__), 'doberman.png')
        image_array = np.asarray(PIL.Image.open(test_image)) / 127.5 - 1.0
        image = image_array.transpose(2, 0, 1)
        image_flip = np.flip(image, axis=1).copy()
        input_image = torch.from_numpy(image).cuda().float()
        input_image_flip = torch.from_numpy(image_flip).cuda().float()
        self.input_batch = torch.stack([input_image, input_image_flip])
        self.integrated_gradients = IntegratedGradients(self.model)
        self.baseline = torch.zeros(self.input_batch.shape)
        self.tolerace = 1e-05 * 25

    def test_get_saliency(self):
        """Tests output shape and type are correct for single input."""
        gradients = self.integrated_gradients.get_saliency(self.input_batch)
        self.assertEqual(gradients.shape, self.input_batch.shape)
        self.assertIs(type(gradients), np.ndarray)

    def test_get_saliency_target_classes(self):
        """Test default and given values of target classes."""
        gradients_predicted = self.integrated_gradients.get_saliency(
            self.input_batch, target_classes=None, baseline=self.baseline
        )
        self.assertEqual(gradients_predicted.shape, self.input_batch.shape)
        self.assertIs(type(gradients_predicted), np.ndarray)

        gradients_class_100 = self.integrated_gradients.get_saliency(
            self.input_batch, target_classes=[100, 100], baseline=self.baseline
        )
        self.assertEqual(gradients_class_100.shape, self.input_batch.shape)
        self.assertIs(type(gradients_class_100), np.ndarray)

        gradients_class_236 = self.integrated_gradients.get_saliency(
            self.input_batch, target_classes=[236, 236], baseline=self.baseline)
        self.assertEqual(gradients_class_236.shape, self.input_batch.shape)
        self.assertIs(type(gradients_class_236), np.ndarray)

        # Input images predicted classes are 236.
        self.assertFalse(np.allclose(gradients_predicted,
                                     gradients_class_100,
                                     atol=self.tolerace))
        self.assertTrue(np.allclose(gradients_predicted,
                                    gradients_class_236,
                                    atol=self.tolerace))
        self.assertFalse(np.allclose(gradients_class_100,
                                     gradients_class_236,
                                     atol=self.tolerace))

    def test_get_saliency_consistency(self):
        """Tests multiple runs return the same gradients."""
        gradients_one = self.integrated_gradients.get_saliency(
            self.input_batch, baseline=self.baseline
        )
        self.assertEqual(gradients_one.shape, self.input_batch.shape)
        self.assertIs(type(gradients_one), np.ndarray)

        gradients_two = self.integrated_gradients.get_saliency(
            self.input_batch, baseline=self.baseline
        )
        self.assertEqual(gradients_two.shape, self.input_batch.shape)
        self.assertIs(type(gradients_two), np.ndarray)

        self.assertTrue(np.allclose(gradients_one,
                                    gradients_two,
                                    atol=self.tolerace))


if __name__ == '__main__':
    unittest.main()
