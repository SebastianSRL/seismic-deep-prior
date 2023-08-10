from tensorflow.keras import layers
from tensorflow.keras.models import Model


class Baseline(Model):
    def __init__(self, nfilters, ksize, depth):
        super(Baseline, self).__init__()
        self.layers_ = list()
        for i in range(depth):
            self.layers_.append(
                layers.Conv3D(
                    nfilters,
                    ksize,
                    padding="same",
                    activation="relu",
                )
            )
        self.layers_.append(
            layers.Conv3D(1, ksize, padding="same", activation="sigmoid")
        )

    def build(self, input_shape):
        super(Baseline, self).build(input_shape)

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.layers_:
            x = layer(x)
        return x
