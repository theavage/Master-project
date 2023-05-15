"""

This scripts defines helping functions for the NN.

"""

import numpy as np
import torch
from dmipy.core.modeling_framework import MultiCompartmentModel
from dataset import MyDataset
from dmipy.core.acquisition_scheme import acquisition_scheme_from_schemefile
import nibabel as nib

# Makes sure the model runs on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_scheme_values(path_to_acqscheme,long_scheme = False):

    """

    Gets the acquisition scheme values from an acquisition scheme.
    args:   path_to_acquscheme: string with path to the scheme file
            long_scheme: if the schemes are long, seperate npy files should be made with all parameters

    """

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

    # Converting from numpy arrays to tensor
    b_values = torch.FloatTensor(b_values)
    gradient_strength = torch.FloatTensor(gradient_strength)
    gradient_directions = torch.FloatTensor(gradient_directions)
    delta = torch.FloatTensor(delta)
    Delta = torch.FloatTensor(Delta)

    return b_values, gradient_strength, gradient_directions, delta, Delta

def load_data(datapath, mask_path=None):

    """

    With help from dataset.py, this funcions loads the data, and potentially applies mask to ensure
    that all voxels outside of mask (air voxels) are set to zero, for easier handling of the data.
    If data is in vivo MRI data, it flattens it from 4D to 2D data. 

    returns: X_data ready for use in model.

    args:   datapath: path to data
            mask_path: path to potential mask

    """

    if datapath.endswith('npz'):
        data = np.load(datapath)
        data = data['arr_0']
    elif datapath.endswith('npy'):
        data = np.load(datapath)
    elif datapath.endswith('.nii.gz') or datapath.endswith('.nii'):
        data = nib.load(datapath).get_fdata()

        if mask_path is not None:
            img_mask = nib.load(mask_path).get_fdata()
    
            masked = data.copy()
            mask = img_mask == 0

            for i in range(data.shape[-1]):
                masked[:,:,:,i][mask] = 0
            
            data = np.nan_to_num(masked,posinf=1,neginf=0)

            img_mask = np.tile(img_mask[...,np.newaxis],(1,1,1,160))
            img_mask = np.reshape(img_mask,[img_mask.shape[0]*img_mask.shape[1]*img_mask.shape[2],img_mask.shape[3]])

            data = np.reshape(data,[data.shape[0]*data.shape[1]*data.shape[2],data.shape[3]])
            
            
            mask = MyDataset(img_mask)
            X_train = MyDataset(data)

            return X_train, mask
            
    else:
        raise Exception("Wrong dataset format: must be numpy or nifti file")
    
    X_train = MyDataset(data)

    return X_train

def sphere_attenuation(gradient_strength, delta, Delta, radius):

    """
    Calculates the sphere signal attenuation. Function is insipred from DMIPY toolbox: 
    Rutger Fick, Rachid Deriche, & Demian Wassermann. https://github.com/AthenaEPI/dmipy
    
    Changes include to make function capable of working with PyTorch tensors.

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
    water_diffusion_constant=1.2e-9,  # m^2/s
    water_in_axons_diffusion_constant=1.2e-9,  # m^2/s
    naa_in_axons=.00015e-9,  # m^2 / s
    water_gyromagnetic_ratio=267.513e6)  # 1/(sT)

    D = const['water_in_axons_diffusion_constant']
    gamma = const['water_gyromagnetic_ratio']
    radius = radius*1e-6# to meter .detach().numpy() #/ 2

    alpha = SPHERE_TRASCENDENTAL_ROOTS.to(device) / radius
    alpha2 = alpha ** 2
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
        summands.sum(axis=0)
    )

    return E

def sphere_compartment(g, delta, Delta, radius):

    """

    performs sphere_attenuation fuctions on all data. 

    args:   g: gradient directions
            delta: diffusion time
            Delta: diffusion duration
            radius: cell radius

    """

    E_sphere = torch.zeros(len(radius),len(g))

    for i in range(len(radius)):
        E_sphere[i][:] = sphere_attenuation(g, delta, Delta, radius[i])

    return E_sphere.to(device)


def unitsphere2cart_Nd(theta,phi):

    """
    
    Converts 1D unit sphere coordinates to cartesian coordinates.

    args:   unit sphere coordinates theta, phi

    Returns: mu, Nd array of size (..., 2)in cartesian coordinates, as x, y, z = mu_cart

    """

    mu_cart = torch.zeros(3,len(theta),device=device)
    sintheta = torch.sin(theta)
    mu_cart[0,:] = torch.squeeze(sintheta * torch.cos(phi))
    mu_cart[1,:] = torch.squeeze(sintheta * torch.sin(phi))
    mu_cart[2,:] = torch.squeeze(torch.cos(theta))
    return mu_cart

def stick_compartment(b_values, lambda_par,gradient_directions,theta,phi):

    """
    
    Converts the 1D unit wphere coordinates to cartesian coordinates, and
    computed signal from the vascular compartment

    args:   b_values: b-value
            lambda_par: the vascular diffusion constant
            gradient_directions: gradient directions
            theta, phi: unit sphere coordinates 
    
    returns: signal from vascular compartment.
    
    """

    mu_cart = unitsphere2cart_Nd(theta,phi)
    dot = torch.einsum("ij,jk->ki",gradient_directions.to(device), mu_cart.to(device))

    return torch.exp(-b_values.to(device) * lambda_par.to(device) * (dot ** 2))


def fractions_to_1(f_sphere,f_ball,f_stick):

    """
    
    Normalize all three compartmemnts signal so they sum to one.

    args:   f_sphere: Spherical, intracellular volume fraction
            f_ball: Ball, extracellular-extravascular volume fraction
            f_vasc: Stick, vascular volume fraction
        
    returns: the three volume fractions so that f_sphere + f_ball + f_vasc = 1
    """

    fractions = torch.stack((f_sphere, f_ball, f_stick))
    normalized_fractions = torch.nn.functional.normalize(fractions,p=1,dim=0)
    f_sphere,f_ball,f_stick = normalized_fractions[0].unsqueeze(1),normalized_fractions[1].unsqueeze(1),normalized_fractions[2].unsqueeze(1)

    return f_sphere.to(device),f_ball.to(device),f_stick.to(device)

    
