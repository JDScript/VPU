import tensorflow as tf
from pointnet2.tf_ops.grouping.tf_grouping import knn_point_2


@tf.keras.utils.register_keras_serializable()
class DenseConv(tf.keras.layers.Layer):
    """
    It's actually a tf.keras.layers.Conv2D with the option to add
    tf.keras.layers.BatchNormalization after convolution
    """

    def __init__(
        self,
        n=3,
        growth_rate=64,
        k=16,
        **kwargs,
    ):
        super(DenseConv, self).__init__(**kwargs)

        self.n = n
        self.growth_rate = growth_rate
        self.k = k

        self.conv_layers = []
        return

    def build(self, input_shape):
        for i in range(self.n):
            if i == self.n - 1:
                self.conv_layers.append(
                    tf.keras.layers.Conv2D(
                        filters=self.growth_rate,
                        kernel_size=(1, 1),
                        padding="VALID",
                    )
                )
            else:
                self.conv_layers.append(
                    tf.keras.layers.Conv2D(
                        filters=self.growth_rate,
                        kernel_size=(1, 1),
                        padding="VALID",
                        activation="relu",
                    )
                )

        self.built = True

    def call(self, inputs, **kwargs):
        outputs, _ = self.get_edge_feature(inputs, k=self.k)
        for i, layer in enumerate(self.conv_layers):
            if i == 0:
                outputs = tf.concat([
                    layer(outputs),
                    tf.tile(tf.expand_dims(inputs, axis=2), [1, 1, self.k, 1])
                ], axis=-1)
            else:
                outputs = tf.concat([
                    layer(outputs),
                    outputs
                ], axis=-1)
        outputs = tf.reduce_max(outputs, axis=-2)
        return outputs

    @staticmethod
    def get_edge_feature(point_cloud, k=16, idx=None):
        if idx is None:
            _, idx = knn_point_2(k + 1, point_cloud, point_cloud, unique=True, sort=True)
            idx = idx[:, :, 1:, :]

        # [N, P, K, Dim]
        point_cloud_neighbors = tf.gather_nd(point_cloud, idx)
        point_cloud_central = tf.expand_dims(point_cloud, axis=-2)

        point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

        edge_feature = tf.concat(
            [point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)
        return edge_feature, idx

    def get_config(self):
        config = super().get_config()
        config["n"] = self.n
        config["growth_rate"] = self.growth_rate
        config["k"] = self.k
        return config
