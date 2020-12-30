# Tests for interpretability methods.

import numpy as np
import PIL.Image
import torch
import torchvision.models as models
import unittest
from unittest.mock import patch
from vanilla_gradients import VanillaGradients
from integrated_gradients import IntegratedGradients


class TestInterpretabilityMethod(unittest.TestCase):
    
    def setUp(self):
        """Set up tests using Inception V3 and VanillaGradients instance."""
        model = models.__dict__['inception_v3'](pretrained=True).cuda().eval()
        self.image = np.asarray(PIL.Image.open('./doberman.png')) / 127.5 - 1.0
        self.vanilla_gradients = VanillaGradients(model)
        self.tolerace = 1e-05
    
    def test_get_smoothgrad_masks_single_input(self):
        """Test smoothgrad function output."""
        input_batch = torch.from_numpy(self.image.transpose(2, 0, 1)).cuda().float().unsqueeze(0)

        # Test output of smoothgrad function.
        smoothgrad_gradients = self.vanilla_gradients.get_smoothgrad_masks(input_batch)
        self.assertEqual(smoothgrad_gradients.shape, input_batch.shape)
        self.assertIs(type(smoothgrad_gradients), np.ndarray)
        
        # Test smoothgrad gradients are unequal to vanilla gradients.
        gradients = self.vanilla_gradients.get_masks(input_batch)
        self.assertFalse(np.allclose(gradients, smoothgrad_gradients, atol=self.tolerace))
        
        # Test single pass of smoothgrad without noise is equal to vanilla gradients squared.
        with patch('torch.Tensor.normal_') as mocked_normal:
            mocked_normal.return_value = torch.zeros(input_batch.shape) # Set noise to 0.
            equivalent_gradients = self.vanilla_gradients.get_smoothgrad_masks(input_batch,
                                                                               num_samples=1,
                                                                               magnitude=False)
            self.assertTrue(np.allclose(gradients, equivalent_gradients, atol=self.tolerace))
            
    def test_get_smoothgrad_masks_batched_input(self):
        """Test smoothgrad function output on batched inputs."""
        input_image = torch.from_numpy(self.image.transpose(2, 0, 1)).cuda().float()
        input_image_duplicate = torch.from_numpy(self.image.transpose(2, 0, 1)).cuda().float()
        input_image_flipped = torch.from_numpy(np.flip(self.image.transpose(2, 0, 1), axis=1).copy()).cuda().float()
        input_batch = torch.stack([input_image, input_image_duplicate, input_image_flipped])
        
        # Test output of smoothgrad function.
        smoothgrad_gradients = self.vanilla_gradients.get_smoothgrad_masks(input_batch)
        self.assertEqual(smoothgrad_gradients.shape, input_batch.shape)
        self.assertIs(type(smoothgrad_gradients), np.ndarray)
        
        # Test first two gradients are equal and unequal to the flipped input. 
        with patch('torch.Tensor.normal_') as mocked_normal:
            mocked_normal.return_value = torch.zeros(input_batch.shape) # Remove noise to check equality.
            smoothgrad_gradients_no_noise = self.vanilla_gradients.get_smoothgrad_masks(input_batch,
                                                                                        magnitude=False)
            self.assertTrue(np.allclose(smoothgrad_gradients_no_noise[0], smoothgrad_gradients_no_noise[1],
                                        atol=self.tolerace))
            self.assertFalse(np.allclose(smoothgrad_gradients_no_noise[0], smoothgrad_gradients_no_noise[2],
                                         atol=self.tolerace))
            self.assertFalse(np.allclose(smoothgrad_gradients_no_noise[1], smoothgrad_gradients_no_noise[2],
                                         atol=self.tolerace))
        

class TestVanillaGradients(unittest.TestCase):
    
    def setUp(self):
        model = models.__dict__['inception_v3'](pretrained=True).cuda().eval()
        image = np.asarray(PIL.Image.open('./doberman.png')) / 127.5 - 1.0
        input_image = torch.from_numpy(image.transpose(2, 0, 1)).cuda().float()
        input_image_flipped = torch.from_numpy(np.flip(image.transpose(2, 0, 1), axis=1).copy()).cuda().float()
        self.input_batch = torch.stack([input_image, input_image_flipped])
        self.vanilla_gradients = VanillaGradients(model)
        self.tolerace = 1e-05
    
    def test_get_masks(self):
        """Tests output shape and type are correct for single input."""
        gradients = self.vanilla_gradients.get_masks(self.input_batch)
        self.assertEqual(gradients.shape, self.input_batch.shape)
        self.assertIs(type(gradients), np.ndarray)
    
    def test_get_masks_target_classes(self):
        """Test default and given values of target classes."""
        gradients_predicted_class = self.vanilla_gradients.get_masks(self.input_batch, 
                                                                     target_classes=None)
        self.assertEqual(gradients_predicted_class.shape, self.input_batch.shape)
        self.assertIs(type(gradients_predicted_class), np.ndarray)
        
        gradients_class_100 = self.vanilla_gradients.get_masks(self.input_batch, 
                                                               target_classes=[100, 100])
        self.assertEqual(gradients_class_100.shape, self.input_batch.shape)
        self.assertIs(type(gradients_class_100), np.ndarray)
        
        gradients_class_236 = self.vanilla_gradients.get_masks(self.input_batch, 
                                                               target_classes=[236, 236])
        self.assertEqual(gradients_class_236.shape, self.input_batch.shape)
        self.assertIs(type(gradients_class_236), np.ndarray)
        
        # Input images predicted classes are 236.
        self.assertFalse(np.allclose(gradients_predicted_class, gradients_class_100, atol=self.tolerace))
        self.assertTrue(np.allclose(gradients_predicted_class, gradients_class_236, atol=self.tolerace))
        self.assertFalse(np.allclose(gradients_class_100, gradients_class_236, atol=self.tolerace))

    def test_get_masks_consistency(self):
        """Tests multiple runs return the same gradients."""
        gradients_first_call= self.vanilla_gradients.get_masks(self.input_batch)
        self.assertEqual(gradients_first_call.shape, self.input_batch.shape)
        self.assertIs(type(gradients_first_call), np.ndarray)
        
        gradients_second_call= self.vanilla_gradients.get_masks(self.input_batch)
        self.assertEqual(gradients_second_call.shape, self.input_batch.shape)
        self.assertIs(type(gradients_second_call), np.ndarray)
        
        self.assertTrue(np.allclose(gradients_first_call, gradients_second_call, atol=self.tolerace))
        
        
class TestIntegratedGradients(unittest.TestCase):
    
    def setUp(self):
        self.model = models.__dict__['inception_v3'](pretrained=True).cuda().eval()
        image = np.asarray(PIL.Image.open('./doberman.png')) / 127.5 - 1.0
        input_image = torch.from_numpy(image.transpose(2, 0, 1)).cuda().float()
        input_image_flipped = torch.from_numpy(np.flip(image.transpose(2, 0, 1), axis=1).copy()).cuda().float()
        self.input_batch = torch.stack([input_image, input_image_flipped])
        self.integrated_gradients = IntegratedGradients(self.model)
        self.tolerace = 1e-05 * 25
    
    def test_get_masks(self):
        """Tests output shape and type are correct for single input."""
        gradients = self.integrated_gradients.get_masks(self.input_batch)
        self.assertEqual(gradients.shape, self.input_batch.shape)
        self.assertIs(type(gradients), np.ndarray)
    
    def test_get_masks_target_classes(self):
        """Test default and given values of target classes."""
        gradients_predicted_class = self.integrated_gradients.get_masks(self.input_batch, 
                                                                        target_classes=None)
        self.assertEqual(gradients_predicted_class.shape, self.input_batch.shape)
        self.assertIs(type(gradients_predicted_class), np.ndarray)
        
        gradients_class_100 = self.integrated_gradients.get_masks(self.input_batch, 
                                                                  target_classes=[100, 100])
        self.assertEqual(gradients_class_100.shape, self.input_batch.shape)
        self.assertIs(type(gradients_class_100), np.ndarray)
        
        gradients_class_236 = self.integrated_gradients.get_masks(self.input_batch, 
                                                                  target_classes=[236, 236])
        self.assertEqual(gradients_class_236.shape, self.input_batch.shape)
        self.assertIs(type(gradients_class_236), np.ndarray)
        
        # Input images predicted classes are 236.
        self.assertFalse(np.allclose(gradients_predicted_class, gradients_class_100, atol=self.tolerace))
        self.assertTrue(np.allclose(gradients_predicted_class, gradients_class_236, atol=self.tolerace))
        self.assertFalse(np.allclose(gradients_class_100, gradients_class_236, atol=self.tolerace))

    def test_get_masks_consistency(self):
        """Tests multiple runs return the same gradients."""
        gradients_first_call= self.integrated_gradients.get_masks(self.input_batch)
        self.assertEqual(gradients_first_call.shape, self.input_batch.shape)
        self.assertIs(type(gradients_first_call), np.ndarray)
        
        gradients_second_call= self.integrated_gradients.get_masks(self.input_batch)
        self.assertEqual(gradients_second_call.shape, self.input_batch.shape)
        self.assertIs(type(gradients_second_call), np.ndarray)
        
        self.assertTrue(np.allclose(gradients_first_call, gradients_second_call, atol=self.tolerace))
        
    def test_get_masks_num_points(self):
        """Test number of points pararmeter."""
        vanilla_gradients_equivalent = VanillaGradients(self.model).get_masks(self.input_batch) * self.input_batch.cpu().detach().numpy()
        integrated_gradients_multiple_points = self.integrated_gradients.get_masks(self.input_batch)
        
        # Integrated gradient with a 0 baseline and alphas = [1] is equal to vanilla gradients times the input.
        with patch('torch.linspace') as mocked_linspace:
            mocked_linspace.return_value = torch.ones((1,)) # Remove noise to check equality.
            integrated_gradients_one_point = self.integrated_gradients.get_masks(self.input_batch, num_points=1)
        
        self.assertTrue(np.allclose(vanilla_gradients_equivalent, integrated_gradients_one_point, atol=self.tolerace))
        self.assertFalse(np.allclose(vanilla_gradients_equivalent, integrated_gradients_multiple_points, atol=self.tolerace))
        self.assertFalse(np.allclose(integrated_gradients_one_point, integrated_gradients_multiple_points, atol=self.tolerace))
        

if __name__ == '__main__':
    unittest.main()