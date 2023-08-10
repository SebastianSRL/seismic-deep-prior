import numpy as np
from utils import random_sampling, uniform_sampling


def gen_training_samples(inputs, phi, subrate, subtype):
    """Generate training samples

    Args:
        x (ndarray): input array (H, W, C)
        phi (ndarray): subsampling mask (C,)
        subrate (float): subsampling rate
        subtype (str): subsampling type

    Returns:
        ndarray: subsampled array (H, W, C)
    """
    nchannels = inputs.shape[-1]
    phi = np.setdiff1d(np.arange(nchannels), phi)
    y = inputs[..., phi]
    subsampling = random_sampling if subtype == "random" else uniform_sampling
    x = subsampling(y, subrate)
    return x, y
