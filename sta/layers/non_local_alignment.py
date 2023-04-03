import tensorflow as tf
from layers import Conv1DWithBatchNorm


@tf.keras.utils.register_keras_serializable()
class NonLocalAlignment(tf.keras.layers.Layer):
    def __init__(
        self,
        use_batch_normalization=False,
        batch_norm_decay=0.0,
        c_channel=8,
        two_dimension=48,
        **kwargs,
    ):
        super(NonLocalAlignment, self).__init__(**kwargs)

        self.use_batch_normalization = use_batch_normalization
        self.batch_norm_decay = batch_norm_decay
        self.c_channel = c_channel
        self.two_dimension = two_dimension

        self.g_t_mlp = None
        self.j_t_mlp = None
        self.k_t_mlp = None
        return

    def build(self, input_shape):
        cross_frame_shape = input_shape[0]
        within_frame_shape = input_shape[1]

        self.g_t_mlp = Conv1DWithBatchNorm(
            self.c_channel,
            1,
            padding="VALID",
            input_shape=cross_frame_shape,
            use_batch_normalization=self.use_batch_normalization,
            batch_norm_decay=self.batch_norm_decay,
            activation=None,
            name="G_t_MLP"
        )

        self.j_t_mlp = Conv1DWithBatchNorm(
            self.c_channel,
            1,
            padding="VALID",
            input_shape=cross_frame_shape,
            use_batch_normalization=self.use_batch_normalization,
            batch_norm_decay=self.batch_norm_decay,
            activation=None,
            name="J_t-1_MLP"
        )

        self.k_t_mlp = Conv1DWithBatchNorm(
            self.two_dimension,
            1,
            padding="VALID",
            input_shape=within_frame_shape,
            use_batch_normalization=self.use_batch_normalization,
            batch_norm_decay=self.batch_norm_decay,
            activation=None,
            name="K_t-1_MLP"
        )

        super().build(input_shape=input_shape)

    def call(self, inputs, **kwargs):
        # inputs contains two parts: cross_combined_f (16, 16 * 128, 48), within_combined_f (16, 8 * 128, 48)
        g_t_feature = self.g_t_mlp(inputs[1], **kwargs)
        j_t_feature = self.j_t_mlp(inputs[0], **kwargs)
        k_t_feature = self.k_t_mlp(inputs[0], **kwargs)

        j_t_feature = tf.transpose(j_t_feature, perm=[0, 2, 1])

        w_t = tf.matmul(g_t_feature, j_t_feature)
        w_t = tf.nn.softmax(w_t)

        return tf.matmul(w_t, k_t_feature)

    def get_config(self):
        config = super().get_config()
        config["use_batch_normalization"] = self.use_batch_normalization
        config["batch_norm_decay"] = self.batch_norm_decay
        config["c_channel"] = self.c_channel
        config["two_dimension"] = self.two_dimension

        return config
