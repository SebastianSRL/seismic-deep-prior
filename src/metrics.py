import tensorflow as tf
from utils import log_tf, tofloat

MAX = 1.0


def mse(y_true, y_pred):
    return tf.keras.losses.MSE(y_true, y_pred)


def psnr(y_true, y_pred):
    y_true, y_pred = tofloat(y_true), tofloat(y_pred)
    return tf.image.psnr(y_true, y_pred, max_val=MAX)


def ssim(y_true, y_pred):
    y_true, y_pred = tofloat(y_true), tofloat(y_pred)
    return tf.image.ssim(y_true, y_pred, filter_size=9, max_val=MAX)


def snr(y_true, y_pred):
    y_true, y_pred = tofloat(y_true), tofloat(y_pred)
    return 20 * log_tf(
        tf.norm(tensor=y_true, ord=2) / tf.norm(tensor=(y_true - y_pred), ord=2)
    )


def print_metrics(y_true, y_pred):
    print("MSE: ", mse(y_true, y_pred).numpy().mean())
    print("PSNR: ", psnr(y_true, y_pred).numpy().item())
    print("SSIM: ", ssim(y_true, y_pred).numpy().item())
    print("SNR: ", snr(y_true, y_pred).numpy().item())
