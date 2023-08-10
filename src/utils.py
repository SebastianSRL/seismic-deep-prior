import numpy as np
import tensorflow as tf


def rescale(data):
    """Min-Max scaling of data

    Args:
        data (ndarray): input array

    Returns:
        ndarray: scaled array between 0 and 1
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def log_tf(x, base=10.0):
    """Tensorflow implementation of element-wise log

    Args:
        x (ndarray): Input array
        base (float, optional): Log base. Defaults to 10.

    Returns:
        Tensor: Element-wise log Tensor of x
    """
    return tf.math.log(x) / tf.math.log(base)


def tofloat(x):
    """Convert data to float32

    Args:
        data (Tensor): input array

    Returns:
        Tensor: float32 Tensor
    """
    return tf.cast(x, tf.float32)


def random_sampling(x, rate, testing=False):
    """Randomly subsample data

    Args:
        x (Tensor): input array (H, W, C)
        rate (float): subsampling rate
        seed (int): random seed

    Returns:
        Tensor: subsampled array
    """
    nchannels = x.shape[-1]
    cut = int(nchannels * rate)
    phi = np.random.permutation(nchannels)[:cut]
    x[:, :, phi] = 0
    if testing:
        return x, phi
    return x


def uniform_sampling(x, rate, testing=False):
    """Uniformly subsample data

    Args:
        x (Tensor): input array (H, W, C)
        rate (float): subsampling rate
        seed (int): random seed

    Returns:
        Tensor: subsampled array
    """
    nchannels = x.shape[-1]
    cut = int(nchannels * rate)
    phi = np.arange(nchannels)[:cut]
    x[..., phi] = 0
    if testing:
        return x, phi
    return x
