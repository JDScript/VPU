import tensorflow as tf
from pointnet2.tf_ops.grouping.tf_grouping import knn_point


@tf.keras.utils.register_keras_serializable()
class FeatureCombination(tf.keras.layers.Layer):
    def __init__(
        self,
        knn,
        **kwargs,
    ):
        super(FeatureCombination, self).__init__(**kwargs)

        self.knn = knn
        return

    def call(self, inputs, **kwargs):
        _, knn_output_idx = knn_point(self.knn, inputs[0], inputs[1])
        batch_num = knn_output_idx.shape[0]
        point_num = knn_output_idx.shape[1]

        previous_features = inputs[2]
        current_features = inputs[3]

        knn_output_idx = tf.cast(knn_output_idx, tf.int32)
        knn_output_idx = tf.reshape(knn_output_idx, (-1, self.knn))

        previous_features = tf.reshape(previous_features, (-1, previous_features.shape[-1]))
        current_features = tf.reshape(current_features, (-1, current_features.shape[-1]))

        previous_selected = tf.gather(previous_features, knn_output_idx, batch_dims=0)
        current_selected = tf.tile(tf.expand_dims(current_features, axis=1), [1, self.knn, 1])

        intermediate_f = tf.concat([previous_selected - current_selected, current_selected], axis=-1)
        intermediate_f = tf.reshape(intermediate_f, (batch_num, point_num * self.knn, -1))

        return intermediate_f

    def get_config(self):
        config = super().get_config()
        config["knn"] = self.knn

        return config
