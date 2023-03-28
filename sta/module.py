import tensorflow as tf


class STAModule(tf.keras.Model):
    def __init__(
        self,
        num_of_frames=3,
    ):
        self.num_of_frames = num_of_frames
        return

    def call(self, inputs, training=None, mask=None):
        return

    def get_config(self):
        config = super().get_config()

        return config
