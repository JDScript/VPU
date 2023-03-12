import tensorflow as tf


class PointNetConv2D(tf.keras.layers.Layer):
    """
    PointNet Conv2D implemented as tf.keras.layers.Layer

    It's actually a tf.keras.layers.Conv2D with the option to add
    tf.keras.layers.BatchNormalization after convolution

    However, we have to inherit tf.keras.layers.Layer, so we have
    to use lower level api tf.nn.conv2d
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
        super(PointNetConv2D, self).__init__(**kwargs)

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

            tf.keras.layers.MaxPooling2D

        self.built = True

    def call(self, inputs):
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
            outputs = self.batch_norm(outputs)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        input_height, input_width, input_channels = input_shape[1:]
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.stride
        padding_height, padding_width = self.padding

        output_height = int((input_height - kernel_height + 2 * padding_height) / stride_height + 1)
        output_width = int((input_width - kernel_width + 2 * padding_width) / stride_width + 1)

        return input_shape[0], output_height, output_width, self.filters