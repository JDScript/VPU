import tensorflow as tf
import numpy as np
from .layers import FeatureExtractionMPU
from layers import Conv2DWithBatchNorm
from sta.module import STAModule
from utils.loss import EMD, Combined, CD, Repulsion


class MPU(tf.keras.Model):
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
        super(MPU, self).__init__(**kwargs)
        self.input_dims = input_dims
        self.radius = radius
        self.use_batch_normalization = use_batch_normalization
        self.batch_norm_decay = batch_norm_decay
        self.up_ratio = up_ratio
        self.sta_module = sta_module

        self.expanded_batch_radius = tf.expand_dims(tf.expand_dims(radius, axis=-1), axis=-1)

        self.feature_extraction = None
        self.up_layer_1 = None
        self.up_layer_2 = None
        self.concat_layer_1 = None
        self.concat_layer_2 = None
        return

    @staticmethod
    def _gen_grid(num_grid_point):
        """
        generate unique indicator for duplication based upsampling module.
        output [num_grid_point, 2]
        """
        x = tf.linspace(-0.2, 0.2, num_grid_point)
        x, y = tf.meshgrid(x, x)
        grid = tf.reshape(tf.stack([x, y], axis=-1), [-1, 2])  # [2, 2, 2] -> [4, 2]
        return grid

    def build(self, input_shape):
        self.feature_extraction = FeatureExtractionMPU(
            12,
            sta_module=self.sta_module
        )
        self.up_layer_1 = Conv2DWithBatchNorm(
            128,
            (1, 1),
            padding="VALID",
            strides=(1, 1),
            use_batch_normalization=self.use_batch_normalization,
            batch_norm_decay=self.batch_norm_decay,
            name="up_layer_1"
        )
        self.up_layer_2 = Conv2DWithBatchNorm(
            128,
            (1, 1),
            padding="VALID",
            strides=(1, 1),
            use_batch_normalization=self.use_batch_normalization,
            batch_norm_decay=self.batch_norm_decay,
            name="up_layer_2"
        )
        self.concat_layer_1 = Conv2DWithBatchNorm(
            64,
            (1, 1),
            activation=tf.nn.leaky_relu,
            name="feat_concat_0"
        )
        self.concat_layer_2 = Conv2DWithBatchNorm(
            3,
            (1, 1),
            activation=None,
            weight_decay=0.,
            name="feat_concat_1"
        )
        return

    def call(self, inputs, **kwargs):
        batch_size = self.input_dims[0]
        num_of_point = self.input_dims[-2]

        features = self.feature_extraction(inputs)

        # Upsampling
        grid = self._gen_grid(tf.cast(tf.round(tf.sqrt(tf.constant(self.up_ratio, dtype=tf.float32))), dtype=tf.int32))
        grid = tf.tile(
            tf.expand_dims(grid, 0),
            [batch_size, num_of_point, 1]
        )
        grid = tf.expand_dims(grid * self.expanded_batch_radius, axis=2)

        new_features = tf.reshape(
            tf.tile(
                tf.expand_dims(features, 2),
                [1, 1, self.up_ratio, 1, 1]
            ),
            [batch_size, num_of_point * self.up_ratio, 1, features.shape[-1]]
        )
        new_features = tf.concat([new_features, grid], axis=-1)
        new_features = self.up_layer_1(new_features)
        new_features = self.up_layer_2(new_features)

        # Feature Concat
        new_xyz = self.concat_layer_1(new_features)
        new_xyz = self.concat_layer_2(new_xyz)

        outputs = tf.squeeze(new_xyz, axis=[2])

        if self.sta_module is not None:
            inputs = inputs[:, inputs.shape[1] // 2 + 1, :, :]

        outputs += tf.reshape(
            tf.tile(
                tf.expand_dims(inputs, 2),
                [1, 1, self.up_ratio, 1]
            ),
            [batch_size, num_of_point * self.up_ratio, 3]
        )
        return outputs

    def build_graph(self):
        x = tf.keras.layers.Input(self.input_dims[1::], batch_size=self.input_dims[0])
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    @staticmethod
    def get_model(
        input_dims=(16, 128, 3),
        radius=1.0,
        use_batch_normalization=False,
        batch_norm_decay=0.00001,
        up_ratio=4,
        learning_rate=0.001,
        sta_module: STAModule | None = None,
    ) -> 'MPU':
        model = MPU(
            input_dims=input_dims,
            radius=radius,
            use_batch_normalization=use_batch_normalization,
            batch_norm_decay=batch_norm_decay,
            up_ratio=up_ratio,
            sta_module=sta_module,
        )
        model(np.zeros(input_dims))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=0.9
            ),
            loss=Combined(
                losses=[CD(), Repulsion()],
                losses_weights=(50, 1),
            ),
        )

        return model
