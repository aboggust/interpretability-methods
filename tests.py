# Tests for interpretability methods.

import numpy as np
import PIL.Image
import torch
import torchvision.models as models
import unittest
from unittest.mock import patch
from vanilla_gradients import VanillaGradients


class TestInterpretabilityMethod(unittest.TestCase):
    
    def setUp(self):
        """Set up tests using Inception V3 and VanillaGradients instance."""
        model = models.__dict__['inception_v3'](pretrained=True).cuda().eval()
        image = np.asarray(PIL.Image.open('./doberman.png')) / 127.5 - 1.0
        self.input_instance = torch.from_numpy(image.transpose(2, 0, 1)).cuda().float()
        self.vanilla_gradients = VanillaGradients(model)
        self.tolerace = 1e-05
    
    def test_get_smoothgrad_mask(self):
        """Test smoothgrad function output."""
        # Test output of smoothgrad function.
        smoothgrad_gradients = self.vanilla_gradients.get_smoothgrad_mask(self.input_instance)
        self.assertEqual(smoothgrad_gradients.shape, self.input_instance.shape)
        self.assertIs(type(smoothgrad_gradients), np.ndarray)
        
        # Test smoothgrad gradients are unequal to vanilla gradients.
        gradients = self.vanilla_gradients.get_mask(self.input_instance)
        self.assertFalse(np.allclose(gradients, smoothgrad_gradients, atol=self.tolerace))
        
        # Test single pass of smoothgrad without noise is equal to vanilla gradients squared.
        with patch('torch.Tensor.normal_') as mocked_normal:
            mocked_normal.return_value = torch.zeros(self.input_instance.shape) # Set noise to 0.
            equivalent_gradients = self.vanilla_gradients.get_smoothgrad_mask(self.input_instance,
                                                                              num_samples=1,
                                                                              magnitude=False)
            self.assertTrue(np.allclose(gradients, equivalent_gradients, atol=self.tolerace))


class TestVanillaGradients(unittest.TestCase):
    
    def setUp(self):
        model = models.__dict__['inception_v3'](pretrained=True).cuda().eval()
        image = np.asarray(PIL.Image.open('./doberman.png')) / 127.5 - 1.0
        self.input_instance = torch.from_numpy(image.transpose(2, 0, 1)).cuda().float()
        self.vanilla_gradients = VanillaGradients(model)
        self.tolerace = 1e-05
    
    def test_get_mask(self):
        """Tests output shape and type are correct for single input."""
        gradients = self.vanilla_gradients.get_mask(self.input_instance)
        self.assertEqual(gradients.shape, self.input_instance.shape)
        self.assertIs(type(gradients), np.ndarray)
    
    def test_get_mask_target_classes(self):
        """Test default and given values of target classes."""
        gradients_default_class = self.vanilla_gradients.get_mask(self.input_instance, 
                                                                  target_class=None)
        self.assertEqual(gradients_default_class.shape, self.input_instance.shape)
        self.assertIs(type(gradients_default_class), np.ndarray)
        
        gradients_class_200 = self.vanilla_gradients.get_mask(self.input_instance, 
                                                              target_class=200)
        self.assertEqual(gradients_class_200.shape, self.input_instance.shape)
        self.assertIs(type(gradients_class_200), np.ndarray)
        
        gradients_class_100 = self.vanilla_gradients.get_mask(self.input_instance, 
                                                              target_class=100)
        self.assertEqual(gradients_class_100.shape, self.input_instance.shape)
        self.assertIs(type(gradients_class_100), np.ndarray)
        
        self.assertFalse(np.allclose(gradients_default_class, gradients_class_100, atol=self.tolerace))
        self.assertFalse(np.allclose(gradients_default_class, gradients_class_200, atol=self.tolerace))
        self.assertFalse(np.allclose(gradients_class_100, gradients_class_200, atol=self.tolerace))

    def test_get_mask_consistency(self):
        """Tests multiple runs return the same gradients."""
        gradients_first_call= self.vanilla_gradients.get_mask(self.input_instance)
        self.assertEqual(gradients_first_call.shape, self.input_instance.shape)
        self.assertIs(type(gradients_first_call), np.ndarray)
        
        gradients_second_call= self.vanilla_gradients.get_mask(self.input_instance)
        self.assertEqual(gradients_second_call.shape, self.input_instance.shape)
        self.assertIs(type(gradients_second_call), np.ndarray)
        
        self.assertTrue(np.allclose(gradients_first_call, gradients_second_call, atol=self.tolerace))
        

if __name__ == '__main__':
    unittest.main()