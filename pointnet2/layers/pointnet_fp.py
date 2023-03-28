import tensorflow as tf
from layers.conv2d_with_batch_norm import Conv2DWithBatchNorm
from ..tf_ops.interpolation.tf_interpolate import three_nn, three_interpolate


@tf.keras.utils.register_keras_serializable()
class PointNetFeaturePropagation(tf.keras.layers.Layer):
    def __init__(
        self,
        mlp_output_shape_each_point: list[int],
        use_batch_normalization=False,
        batch_norm_decay=0.00001,
        **kwargs
    ):
        super(PointNetFeaturePropagation, self).__init__(**kwargs)

        self.mlp_output_shape_each_point = mlp_output_shape_each_point
        self.use_batch_normalization = use_batch_normalization
        self.batch_norm_decay = batch_norm_decay

        # Init empty layers
        self.mlp_layers_each_point = []

    def build(self, input_shape):
        self.mlp_layers_each_point = [
            Conv2DWithBatchNorm(
                num_out_channel,
                (1, 1),
                padding="VALID",
                use_batch_normalization=self.use_batch_normalization,
                batch_norm_decay=self.batch_norm_decay,
                strides=(1, 1),
                name=f"MLP_each_point_{i}"
            )
            for i, num_out_channel in enumerate(self.mlp_output_shape_each_point)
        ]
        super().build(input_shape=input_shape)

    def call(self, inputs):
        xyz1, xyz2, points1, points2 = inputs

        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0 / dist), axis=2, keepdims=True)
        norm = tf.tile(norm, [1, 1, 3])
        weight = (1.0 / dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1])
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)

        for layer in self.mlp_layers_each_point:
            new_points1 = layer(new_points1)

        new_points1 = tf.squeeze(new_points1, axis=[2])
        return new_points1

    def get_config(self):
        config = super().get_config()
        config["mlp_output_shape_each_point"] = self.mlp_output_shape_each_point
        config["use_batch_normalization"] = self.use_batch_normalization
        config["batch_norm_decay"] = self.batch_norm_decay
        return config
