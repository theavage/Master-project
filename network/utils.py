import numpy as np
import torch
from dmipy.signal_models import sphere_models, cylinder_models, gaussian_models
from dmipy.core.modeling_framework import MultiCompartmentModel, MultiCompartmentSphericalHarmonicsModel

def squash(param, p_min, p_max):
    """

    torch.clamp Clamps all elements in input into the range [ min, max ],
    before tensor.unsqueeze returns a new tensor with a dimension of size one inserted at the specified position.

    """
    squashed_param_tensor =torch.clamp(param, min=p_min, max=p_max)
    unsqueezed_param = squashed_param_tensor.unsqueeze(1)
    return unsqueezed_param

def verdict_model_dmipy():


    sphere = sphere_models.S4SphereGaussianPhaseApproximation(diffusion_constant=1.2e-9)
    ball = gaussian_models.G1Ball()
    stick = cylinder_models.C1Stick()

    verdict_mod = MultiCompartmentModel(models=[sphere, ball, stick])

    verdict_mod.set_fixed_parameter('G1Ball_1_lambda_iso', 2e-9) #2
    verdict_mod.set_parameter_optimization_bounds('C1Stick_1_lambda_par', [3.05e-9, 10e-9])

