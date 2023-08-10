import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from default import get_default_config
from metrics import print_metrics, psnr, snr, ssim
from preprocessing import gen_training_samples
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from utils import random_sampling, rescale, uniform_sampling

from models import Baseline

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))


def parser():
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, default="src/config.yaml")
    return args.parse_args()


def main(cfg):
    np.random.seed(cfg.TRAIN.seed)
    tf.random.set_seed(cfg.TRAIN.seed)

    y_test = np.load(cfg.DATASET.data_path)
    y_test = rescale(y_test)

    if cfg.DATASET.phi is not None:
        x_test = y_test.copy()
        phi = cfg.DATASET.phi
        x_test[:, phi] = 0
    else:
        subsampling = (
            random_sampling if cfg.TRAIN.subtype == "random" else uniform_sampling
        )
        x_test, phi = subsampling(y_test, cfg.TRAIN.subrate, testing=True)

    x_train, y_train = gen_training_samples(
        x_test, phi, cfg.TRAIN.subrate, cfg.TRAIN.subtype
    )

    x_train = np.expand_dims(x_train, [0, -1])
    y_train = np.expand_dims(y_train, [0, -1])
    x_test = np.expand_dims(x_test, [0, -1])
    y_test = np.expand_dims(y_test, [0, -1])

    if not Path(cfg.TRAIN.log_path).exists():
        Path(cfg.TRAIN.log_path).parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            cfg.MODEL.save_path,
            save_weights_only=True,
            save_best_only=True,
            monitor="val_psnr",
        ),
        CSVLogger(
            cfg.TRAIN.log_path,
        ),
    ]

    model = Baseline(cfg.MODEL.nfilters, cfg.MODEL.ksize, cfg.MODEL.depth)
    model.compile(
        optimizer=Adam(cfg.TRAIN.lr),
        loss="mse",
        metrics=["mse", "mae", psnr, ssim, snr],
    )
    model.build(input_shape=(None, None, None, None, 1))
    model.summary()
    model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=callbacks)
    model.load_weights(cfg.MODEL.save_path)
    y_pred = model.predict(x_test)

    y_pred = np.squeeze(y_pred, -1)[..., phi]
    y_test = np.squeeze(y_test, -1)[..., phi]
    print_metrics(y_test, y_pred)


if __name__ == "__main__":
    args = parser()
    cfg = get_default_config()
    cfg.merge_from_file(args.config)
    cfg.freeze()
    main(cfg)
