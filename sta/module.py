import tensorflow as tf
from layers import Conv1DWithBatchNorm, Conv2DWithBatchNorm
from .layers.feature_combination import FeatureCombination
from .layers.non_local_alignment import NonLocalAlignment


class STAModule(tf.keras.Model):
    def __init__(
        self,
        num_of_frames=3,
        use_batch_normalization=False,
        batch_norm_decay=0.0,
        knn=8,
        dimension=24,
        **kwargs,
    ):
        super(STAModule, self).__init__(**kwargs)
        self.use_batch_normalization = use_batch_normalization
        self.batch_norm_decay = batch_norm_decay
        self.knn = knn
        self.dimension = dimension

        self.num_of_frames = num_of_frames
        self.shared_mlp = None
        self.cross_combined_submodule = None
        self.within_combined_submodule = None
        self.previous_non_local_alignment = None
        self.next_non_local_alignment = None
        self.final_mlp = None

        return

    def build(self, input_shape):
        self.shared_mlp = Conv1DWithBatchNorm(
            24,
            1,
            padding="VALID",
            input_shape=(input_shape[0], input_shape[2], input_shape[3]),
            use_batch_normalization=self.use_batch_normalization,
            batch_norm_decay=self.batch_norm_decay,
            activation=None,
            name="STA_Shared_MLP"
        )

        self.cross_combined_submodule = FeatureCombination(knn=16, name="CFFC_Submodule")
        self.within_combined_submodule = FeatureCombination(knn=8, name="WFFC_Submodule")

        self.previous_non_local_alignment = NonLocalAlignment(
            use_batch_normalization=False,
            batch_norm_decay=0.0,
            c_channel=8,
            two_dimension=48,
            name="previous_non_local_alignment_submodule"
        )

        self.next_non_local_alignment = NonLocalAlignment(
            use_batch_normalization=False,
            batch_norm_decay=0.0,
            c_channel=8,
            two_dimension=48,
            name="next_non_local_alignment_submodule"
        )

        self.final_mlp = Conv1DWithBatchNorm(
            24,
            1,
            padding="VALID",
            input_shape=(input_shape[0], input_shape[2], 6*self.dimension),
            use_batch_normalization=self.use_batch_normalization,
            batch_norm_decay=self.batch_norm_decay,
            activation=None,
            name="STA_final_MLP"
        )

        super().build(input_shape=input_shape)

    @tf.function
    def call(self, inputs, **kwargs):
        feature_tensors = []
        frames = inputs.shape[1]
        for i in range(frames):
            frame = inputs[:, i, :, :]
            feature_tensor = self.shared_mlp(frame, **kwargs)
            feature_tensors.append(feature_tensor)

        features = tf.stack(feature_tensors, axis=1)
        previous_combined_f = self.cross_combined_submodule([inputs[:, -1, :, :],
                                                             inputs[:, 0, :, :],
                                                             features[:, -1, :, :],
                                                             features[:, 0, :, :]])
        within_combined_f = self.within_combined_submodule([inputs[:, 0, :, :],
                                                            inputs[:, 0, :, :],
                                                            features[:, 0, :, :],
                                                            features[:, 0, :, :]])
        next_combined_f = self.cross_combined_submodule([inputs[:, 1, :, :],
                                                         inputs[:, 0, :, :],
                                                         features[:, 1, :, :],
                                                         features[:, 0, :, :]])

        previous_transformed_f = self.previous_non_local_alignment([previous_combined_f, within_combined_f])
        next_transformed_f = self.next_non_local_alignment([next_combined_f, within_combined_f])

        combined_f = tf.stack([within_combined_f, previous_transformed_f, next_transformed_f], axis=0)
        combined_f = tf.reshape(combined_f, (inputs.shape[0], inputs.shape[2], self.knn, 6*self.dimension))

        max_pooling_f = tf.reduce_max(combined_f, axis=2)

        aggregated_f = self.final_mlp(max_pooling_f)

        return aggregated_f

    def build_graph(self):
        x = tf.keras.layers.Input((3, 128, 3), batch_size=16)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def get_config(self):
        config = super().get_config()

        return config
