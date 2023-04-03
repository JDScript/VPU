import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class Conv2DWithBatchNorm(tf.keras.layers.Layer):
    """
    It's actually a tf.keras.layers.Conv2D with the option to add
    tf.keras.layers.BatchNormalization after convolution
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="SAME",
        initializer="glorot_uniform",
        activation=tf.nn.relu,
        weight_decay=0.00001,
        use_batch_normalization=False,
        batch_norm_decay=0.00001,
        **kwargs,
    ):
        super(Conv2DWithBatchNorm, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.initializer = initializer
        self.activation = activation
        self.weight_decay = weight_decay
        self.use_batch_normalization = use_batch_normalization
        self.batch_norm_decay = batch_norm_decay

        self.kernel = None
        self.bias = None
        self.batch_norm = None
        return

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        kernel_shape = self.kernel_size + (
            input_shape[-1],
            self.filters,
        )

        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.initializer,
            regularizer=tf.keras.regularizers.l2(self.weight_decay),
            trainable=True
        )

        self.bias = self.add_weight(
            name="bias",
            shape=(self.filters,),
            initializer="zeros",
            trainable=True
        )

        if self.use_batch_normalization:
            self.batch_norm = tf.keras.layers.BatchNormalization(
                momentum=self.batch_norm_decay,
                fused=True,
                renorm=False,
            )

        self.built = True

    def call(self, inputs, **kwargs):
        outputs = tf.nn.conv2d(
            inputs,
            filters=self.kernel,
            strides=self.strides,
            padding=self.padding,
        )

        outputs = tf.nn.bias_add(
            outputs,
            self.bias,
        )

        if self.batch_norm is not None:
            outputs = self.batch_norm(outputs, **kwargs)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def get_config(self):
        config = super().get_config()
        config["filters"] = self.filters
        config["kernel_size"] = self.kernel_size
        config["strides"] = self.strides
        config["padding"] = self.padding
        config["initializer"] = self.initializer
        config["activation"] = self.activation
        config["weight_decay"] = self.weight_decay
        config["use_batch_normalization"] = self.use_batch_normalization
        config["batch_norm_decay"] = self.batch_norm_decay

        return config

