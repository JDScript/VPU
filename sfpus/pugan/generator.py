import tensorflow as tf

from sfpus.mpu.layers import FeatureExtractionMPU
from sta.module import STAModule


class PUGAN(tf.keras.Model):
    def __init__(
        self,
        input_dims=(16, 128, 3),
        radius=1.0,
        use_batch_normalization=False,
        batch_norm_decay=0.00001,
        up_ratio=4,
        sta_module: STAModule | None = None,
        **kwargs,
    ):
        super(PUGAN, self).__init__(**kwargs)
        self.input_dims = input_dims
        self.radius = radius
        self.use_batch_normalization = use_batch_normalization
        self.batch_norm_decay = batch_norm_decay
        self.up_ratio = up_ratio
        self.sta_module = sta_module

        self.feature_extraction = None


    def build(self, input_shape):
        self.feature_extraction = FeatureExtractionMPU(
            24,
            sta_module=self.sta_module
        )
        return

    def call(self, inputs, training=None, mask=None):
        features = self.feature_extraction(inputs)