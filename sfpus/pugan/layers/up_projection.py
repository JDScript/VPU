import tensorflow as tf


class UpProjection(tf.keras.layers.Layer):
    def __init__(
        self,
        up_ratio,
        use_batch_normalization=False,
        batch_norm_decay=0.00001,
        **kwargs,
    ):
        super(UpProjection, self).__init__(**kwargs)
