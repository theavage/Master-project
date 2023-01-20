import numpy as np
import torch
from dmipy.signal_models import sphere_models, cylinder_models, gaussian_models
from dmipy.core.modeling_framework import MultiCompartmentModel
from dataset import MyDataset
from dmipy.core.acquisition_scheme import acquisition_scheme_from_schemefile

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

    return verdict_mod

def get_scheme_values(path_to_acqscheme,long_scheme = False):

    if long_scheme == False:
        scheme = acquisition_scheme_from_schemefile(path_to_acqscheme)

        b_values = scheme.bvalues
        gradient_strength = scheme.gradient_strengths
        gradient_directions = scheme.gradient_directions
        delta = scheme.delta
        Delta = scheme.Delta

    else:

        b_values = np.load('data/bvalues.npy')
        gradient_strength = np.load('data/G.npy')
        gradient_directions = np.load('data/gradient_directions.npy')
        delta = np.load('data/d.npy')
        Delta = np.load('data/Delta.npy')

    b_values = torch.FloatTensor(b_values)
    gradient_strength = torch.FloatTensor(gradient_strength)
    gradient_directions = torch.FloatTensor(gradient_directions)
    delta = torch.FloatTensor(delta)
    Delta = torch.FloatTensor(Delta)
    return b_values, gradient_strength, gradient_directions, delta, Delta

def load_data(datapath):

    if datapath.endswith('npz'):
        data = np.load(datapath)
        data = data['arr_0']
    elif datapath.endswith('npy'):
        data = np.load(datapath)
    else:
        raise Exception("Wrong dataset format: must be numpy file")
    
    X_train = MyDataset(data)

    return X_train

def sphere_attenuation(gradient_strength, delta, Delta, radius):
    """
    Calculates the sphere signal attenuation.
    From DMIPY
    """
    SPHERE_TRASCENDENTAL_ROOTS = torch.FloatTensor([
        # 0.,
        2.081575978, 5.940369990, 9.205840145,
        12.40444502, 15.57923641, 18.74264558, 21.89969648,
        25.05282528, 28.20336100, 31.35209173, 34.49951492,
        37.64596032, 40.79165523, 43.93676147, 47.08139741,
        50.22565165, 53.36959180, 56.51327045, 59.65672900,
        62.80000055, 65.94311190, 69.08608495, 72.22893775,
        75.37168540, 78.51434055, 81.65691380, 84.79941440,
        87.94185005, 91.08422750, 94.22655255, 97.36883035,
        100.5110653, 103.6532613, 106.7954217, 109.9375497,
        113.0796480, 116.2217188, 119.3637645, 122.5057870,
        125.6477880, 128.7897690, 131.9317315, 135.0736768,
        138.2156061, 141.3575204, 144.4994207, 147.6413080,
        150.7831829, 153.9250463, 157.0668989, 160.2087413,
        163.3505741, 166.4923978, 169.6342129, 172.7760200,
        175.9178194, 179.0596116, 182.2013968, 185.3431756,
        188.4849481, 191.6267147, 194.7684757, 197.9102314,
        201.0519820, 204.1937277, 207.3354688, 210.4772054,
        213.6189378, 216.7606662, 219.9023907, 223.0441114,
        226.1858287, 229.3275425, 232.4692530, 235.6109603,
        238.7526647, 241.8943662, 245.0360648, 248.1777608,
        251.3194542, 254.4611451, 257.6028336, 260.7445198,
        263.8862038, 267.0278856, 270.1695654, 273.3112431,
        276.4529189, 279.5945929, 282.7362650, 285.8779354,
        289.0196041, 292.1612712, 295.3029367, 298.4446006,
        301.5862631, 304.7279241, 307.8695837, 311.0112420,
        314.1528990
    ])

    SPHERE_TRASCENDENTAL_ROOTS =  SPHERE_TRASCENDENTAL_ROOTS.tile(len(Delta),1).T

    const = dict(
    water_diffusion_constant=2.299e-9,  # m^2/s
    water_in_axons_diffusion_constant=1.7e-9,  # m^2/s
    naa_in_axons=.00015e-9,  # m^2 / s
    water_gyromagnetic_ratio=267.513e6)  # 1/(sT)

    D = const['water_in_axons_diffusion_constant']
    gamma = const['water_gyromagnetic_ratio']
    radius = radius*1e-6# to meter .detach().numpy() #/ 2

    alpha = SPHERE_TRASCENDENTAL_ROOTS / radius
    alpha2 = torch.FloatTensor(alpha ** 2)
    alpha2D = alpha2 * D


    first_factor = -2 * (gamma * gradient_strength) ** 2 / D
    summands = (
        alpha ** (-4) / (alpha2 * radius ** 2 - 2) *
        (
            2 * delta - (
                2 +
                torch.exp(-alpha2D * (Delta - delta)) -
                2 * torch.exp(-alpha2D * delta) -
                2 * torch.exp(-alpha2D * Delta) +
                torch.exp(-alpha2D * (Delta + delta))
            ) / (alpha2D)
        )
    )
    E = torch.exp(
        first_factor *
        summands.sum()
    )

    return E

def sphere_compartment(g, delta, Delta, radius):
    """
    E_sphere = torch.zeros(32,len(g)).cuda()
    # for every unique combination get the perpendicular attenuation
    
    for i in range(len(radius)):
        for j in range(len(g)):
            E_sphere[i,j] = sphere_attenuation(g[j],delta[j],Delta[j],radius[i])
    
    return E_sphere.to(torch.device("cpu"))
    """

    E_sphere = torch.zeros(len(radius),len(g))

    for i in range(len(radius)):
        E_sphere[i][:] = sphere_attenuation(g, delta, Delta, radius[i])

    return E_sphere


def unitsphere2cart_Nd(theta,phi):
    """Optimized function deicated to convert 1D unit sphere coordinates
    to cartesian coordinates.
    Parameters
    ----------
    mu : Nd array of size (..., 2)
        unit sphere coordinates, as theta, phi = mu
    Returns
    -------
    mu_cart, Nd array of size (..., 3)
        mu in cartesian coordinates, as x, y, z = mu_cart
"""
    mu_cart = torch.zeros(3,len(theta),device=torch.device("cpu"))
    sintheta = torch.sin(theta)
    mu_cart[0,:] = torch.squeeze(sintheta * torch.cos(phi))
    mu_cart[1,:] = torch.squeeze(sintheta * torch.sin(phi))
    mu_cart[2,:] = torch.squeeze(torch.cos(theta))
    return mu_cart

def stick_compartment(b_values, lambda_par,gradient_directions,theta,phi):
    mu_cart = unitsphere2cart_Nd(theta,phi)
    dot = torch.einsum("ij,jk->ki",gradient_directions, mu_cart)
    return torch.exp(-b_values * lambda_par.to(torch.device("cpu")) * dot ** 2)

def fractions_to_1(f_sphere,f_ball,f_stick):

    m = torch.nn.Softmax(dim = 0)
    volume_fractions = torch.stack((f_sphere, f_ball, f_stick))
    normalized_fractions = m(volume_fractions)
    f_sphere,f_ball,f_stick = normalized_fractions[0].unsqueeze(1),normalized_fractions[1].unsqueeze(1),normalized_fractions[2].unsqueeze(1)
    device = torch.device("cpu")

    return f_sphere.to(device),f_ball.to(device),f_stick.to(device)

    
