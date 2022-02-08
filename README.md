# Interpretability Methods
This repository contains implementations of common attribution-based interpretability methods. See the [example notebook](https://github.mit.edu/aboggust/interpretability_methods/blob/master/examples/interpretability_examples.ipynb)!

Current methods that are implemented:
* **Vanilla Gradients** ([paper 1](https://www.researchgate.net/profile/Aaron_Courville/publication/265022827_Visualizing_Higher-Layer_Features_of_a_Deep_Network/links/53ff82b00cf24c81027da530.pdf) | [paper 2](https://arxiv.org/pdf/1312.6034.pdf) | implemented via [Captum](https://captum.ai/api/saliency.html))
* **Input x Gradient** ([paper](https://arxiv.org/pdf/1605.01713.pdf) | implemented via [Captum](https://captum.ai/api/input_x_gradient.html))
* **Integrated Gradients** ([paper](https://arxiv.org/pdf/1703.01365.pdf) | implemented via [Captum](https://captum.ai/api/integrated_gradients.html))
* **SmoothGrad** ([paper](https://arxiv.org/abs/1706.03825.pdf) | adapted from [Google Pair Saliency](https://github.com/PAIR-code/saliency))
* **Guided Backprop** ([paper](https://arxiv.org/pdf/1412.6806.pdf) | implemented via [Captum](https://captum.ai/api/guided_backprop.html))
* **GradCAM** ([paper](https://arxiv.org/pdf/1610.02391.pdf) | implemented via [Captum](https://captum.ai/api/layer.html#gradcam))
* **Guided GradCAM** ([paper](https://arxiv.org/pdf/1610.02391.pdf) | implemented via [Captum](https://captum.ai/api/guided_grad_cam.html))
* **XRAI** ([paper](https://arxiv.org/pdf/1906.02825.pdf) | adapted from [Google Pair Saliency](https://github.com/PAIR-code/saliency))
* **LIME** ([paper](https://arxiv.org/pdf/1602.04938.pdf) | implemented via the [author's repo](https://github.com/marcotcr/lime))

Each method performs batched computation and can be computed with and without SmoothGrad. The methods are implemented using [Captum](https://captum.ai/) and puplic repostiories (i.e., [LIME](https://github.com/marcotcr/lime)) and are largley inspired by the [Google Pair Saliency implementation](https://github.com/PAIR-code/saliency).

## Usage
### Step 1: Install interpretability_methods.
Install the method locally for use in other development projects. It can be referenced as `interpretability_methods` within this package and in other locations.  
```pip install git+https://github.com/aboggust/interpretability-methods.git```

### Step 2: Install the requirements.
Requirements are listed in [`requirements.txt`](https://github.mit.edu/aboggust/interpretability_methods/blob/master/requirements.txt). Install via:  
```pip install -r requirements.txt```

### Step 3: Produce saliency.
See [notebook](https://github.mit.edu/aboggust/interpretability_methods/blob/master/examples/interpretability_examples.ipynb) for examples.

Each interpretability method (i.e., `VanillaGradients`) extends the base class `InterpretabilityMethod`. Each method is instantiated with a model and, optionally, other method specific parameters. An `InterpretabilityMethod` object has two public methods: `get_saliency` and `get_saliency_smoothed`. 

`get_saliency` takes in an `input_batch` (e.g., a batch of images) and outputs an `np.array` of the same size that represents the attributions. It defaults to computing the saliency with respect the the model's predicted class, but `target_classes` can optionally be passed to specify a specific class. `target_classes` is a list of integers the same length as the batch size. `target_class[i]` is the index of the class to compute saliency with respect to for `input_batch[i]`.

`get_saliency_smoothed` applies SmoothGrad to the `get_saliency` attributions.

Once saliency is computed, [`util.py`](https://github.mit.edu/aboggust/interpretability_methods/blob/master/util.py) contains code to visualize the attributions.

Usage example:
```
# Getting Vanilla Gradients with respect to the predicted class.
from interpretability_methods.vanilla_gradients import VanillaGradients
from interpretability_methods.util import visualize_saliency

model = ... # assuming pytorch model 
input_batch = ... # assumping 4D input batch (batch, channels, height, width)
vanilla_gradients_method = VanillaGradients(model)
vanilla_gradients = vanilla_gradients_method(input_batch) # attributions of shape (batch, channels, height, width)
visualize_saliency(vanilla_gradients) # will output greyscale saliency image
```
