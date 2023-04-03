import tensorflow as tf
from sta.module import STAModule
from layers import Conv1DWithBatchNorm, Conv2DWithBatchNorm, DenseConv


@tf.keras.utils.register_keras_serializable()
class FeatureExtractionMPU(tf.keras.layers.Layer):
    def __init__(
        self,
        growth_rate=12,
        dense_n=3,
        knn=16,
        use_batch_normalization=False,
        batch_norm_decay=0.00001,
        sta_module: STAModule | None = None,
        **kwargs,
    ):
        super(FeatureExtractionMPU, self).__init__(**kwargs)
        self.growth_rate = growth_rate
        self.dense_n = dense_n
        self.knn = knn
        self.use_batch_normalization = use_batch_normalization
        self.batch_norm_decay = batch_norm_decay
        self.sta_module = sta_module

        self.layer0 = None
        self.layer1_dense_conv = None
        self.layer2_conv1d = None
        self.layer2_dense_conv = None
        self.layer3_conv1d = None
        self.layer3_dense_conv = None
        self.layer4_conv1d = None
        self.layer4_dense_conv = None

    def call(self, inputs, *args, **kwargs):
        if self.sta_module is None:
            l0_features = tf.expand_dims(inputs, axis=2)
            l0_features = self.layer0(l0_features)
            l0_features = tf.squeeze(l0_features, axis=2)
        else:
            l0_features = self.sta_module(inputs)

        # Encoding
        l1_features = self.layer1_dense_conv(l0_features)
        l1_features = tf.concat([l1_features, l0_features], axis=-1)

        l2_features = self.layer2_conv1d(l1_features)
        l2_features = self.layer2_dense_conv(l2_features)
        l2_features = tf.concat([l2_features, l1_features], axis=-1)

        l3_features = self.layer3_conv1d(l2_features)
        l3_features = self.layer3_dense_conv(l3_features)
        l3_features = tf.concat([l3_features, l2_features], axis=-1)

        l4_features = self.layer4_conv1d(l3_features)
        l4_features = self.layer4_dense_conv(l4_features)
        l4_features = tf.concat([l4_features, l3_features], axis=-1)

        l4_features = tf.expand_dims(l4_features, axis=2)

        return l4_features

    def build(self, input_shape):
        self.layer0 = Conv2DWithBatchNorm(
            24,
            (1, 1),
            padding="VALID",
            use_batch_normalization=self.use_batch_normalization,
            batch_norm_decay=self.batch_norm_decay,
            activation=None,
        )

        self.layer1_dense_conv = DenseConv(
            growth_rate=self.growth_rate,
            k=self.knn,
            n=self.dense_n,
            name="layer_1"
        )
        self.layer2_dense_conv = DenseConv(
            growth_rate=self.growth_rate,
            k=self.knn,
            n=self.dense_n,
            name="layer_2"
        )
        self.layer3_dense_conv = DenseConv(
            growth_rate=self.growth_rate,
            k=self.knn,
            n=self.dense_n,
            name="layer_3"
        )
        self.layer4_dense_conv = DenseConv(
            growth_rate=self.growth_rate,
            k=self.knn,
            n=self.dense_n,
            name="layer_4"
        )
        self.layer2_conv1d = Conv1DWithBatchNorm(
            self.growth_rate * 2,
            1,
            padding="VALID",
            use_batch_normalization=self.use_batch_normalization,
            batch_norm_decay=self.batch_norm_decay,
            name="layer_2_prep"
        )
        self.layer3_conv1d = Conv1DWithBatchNorm(
            self.growth_rate * 2,
            1,
            padding="VALID",
            use_batch_normalization=self.use_batch_normalization,
            batch_norm_decay=self.batch_norm_decay,
            name="layer_3_prep"
        )
        self.layer4_conv1d = Conv1DWithBatchNorm(
            self.growth_rate * 2,
            1,
            padding="VALID",
            use_batch_normalization=self.use_batch_normalization,
            batch_norm_decay=self.batch_norm_decay,
            name="layer_4_prep"
        )
        return

    def get_config(self):
        config = super().get_config()
        config["growth_rate"] = self.growth_rate
        config["dense_n"] = self.dense_n
        config["knn"] = self.knn
        config["use_batch_normalization"] = self.use_batch_normalization
        config["batch_norm_decay"] = self.batch_norm_decay
        return config
