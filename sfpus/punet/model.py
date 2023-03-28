import tensorflow as tf
import numpy as np
from pointnet2.layers import PointNetSetAbstraction, PointNetFeaturePropagation
from layers.conv2d_with_batch_norm import Conv2DWithBatchNorm
from utils.loss import EMD, Repulsion, Combined


class PUNet(tf.keras.Model):
    def __init__(
        self,
        input_dims=(16, 128, 3),
        radius=1.0,
        use_batch_normalization=False,
        batch_norm_decay=0.00001,
        up_ratio=4,
        **kwargs
    ):
        super(PUNet, self).__init__(**kwargs)
        self.input_dims = input_dims
        self.radius = radius
        self.use_batch_normalization = use_batch_normalization
        self.batch_norm_decay = batch_norm_decay
        self.up_ratio = up_ratio

        self.pointnet_sa_1 = PointNetSetAbstraction(
            name="layer1_sa",
            num_of_fps_points=input_dims[1],
            radius=self.radius * 0.05,
            num_of_local_points=32,
            use_batch_normalization=self.use_batch_normalization,
            batch_norm_decay=self.batch_norm_decay,
            mlp_output_shape_each_point=(32, 32, 64),
            mlp_output_shape_each_region=None,
            group_all=False,
        )

        self.pointnet_sa_2 = PointNetSetAbstraction(
            name="layer2_sa",
            num_of_fps_points=input_dims[1] // 2,
            radius=self.radius * 0.1,
            num_of_local_points=32,
            use_batch_normalization=self.use_batch_normalization,
            batch_norm_decay=self.batch_norm_decay,
            mlp_output_shape_each_point=(64, 64, 128),
            mlp_output_shape_each_region=None,
            group_all=False,
        )

        self.pointnet_sa_3 = PointNetSetAbstraction(
            name="layer3_sa",
            num_of_fps_points=input_dims[1] // 4,
            radius=self.radius * 0.2,
            num_of_local_points=32,
            use_batch_normalization=self.use_batch_normalization,
            batch_norm_decay=self.batch_norm_decay,
            mlp_output_shape_each_point=(128, 128, 256),
            mlp_output_shape_each_region=None,
            group_all=False,
        )

        self.pointnet_sa_4 = PointNetSetAbstraction(
            name="layer4_sa",
            num_of_fps_points=input_dims[1] // 8,
            radius=self.radius * 0.3,
            num_of_local_points=32,
            use_batch_normalization=self.use_batch_normalization,
            batch_norm_decay=self.batch_norm_decay,
            mlp_output_shape_each_point=(256, 256, 512),
            mlp_output_shape_each_region=None,
            group_all=False,
        )

        self.pointnet_fp_1 = PointNetFeaturePropagation(
            name="up_l4",
            use_batch_normalization=self.use_batch_normalization,
            batch_norm_decay=self.batch_norm_decay,
            mlp_output_shape_each_point=[64]
        )

        self.pointnet_fp_2 = PointNetFeaturePropagation(
            name="up_l3",
            use_batch_normalization=self.use_batch_normalization,
            batch_norm_decay=self.batch_norm_decay,
            mlp_output_shape_each_point=[64]
        )

        self.pointnet_fp_3 = PointNetFeaturePropagation(
            name="up_l2",
            use_batch_normalization=self.use_batch_normalization,
            batch_norm_decay=self.batch_norm_decay,
            mlp_output_shape_each_point=[64]
        )

        self.up_layers = [
            [
                Conv2DWithBatchNorm(256, (1, 1), use_batch_normalization=self.use_batch_normalization,
                                    batch_norm_decay=self.batch_norm_decay, name=f"feat_concat_{i}_0"),
                Conv2DWithBatchNorm(128, (1, 1), use_batch_normalization=self.use_batch_normalization,
                                    batch_norm_decay=self.batch_norm_decay, name=f"feat_concat_{i}_1"),
            ]
            for i in range(self.up_ratio)
        ]

        self.conv1 = Conv2DWithBatchNorm(64, (1, 1), activation=tf.nn.leaky_relu, name="final_feat_concat_0")
        self.conv2 = Conv2DWithBatchNorm(3, (1, 1), activation=None, weight_decay=0., name="final_feat_concat_1")
        return

    def call(self, inputs, **kwargs):
        l0_xyz = inputs[:, :, 0:3]
        l0_normals = None

        l1_xyz, l1_points, _ = self.pointnet_sa_1((l0_xyz, l0_normals))
        l2_xyz, l2_points, _ = self.pointnet_sa_2((l1_xyz, l1_points))
        l3_xyz, l3_points, _ = self.pointnet_sa_3((l2_xyz, l2_points))
        l4_xyz, l4_points, _ = self.pointnet_sa_4((l3_xyz, l3_points))

        up_l4_points = self.pointnet_fp_1((l0_xyz, l4_xyz, None, l4_points))
        up_l3_points = self.pointnet_fp_2((l0_xyz, l3_xyz, None, l3_points))
        up_l2_points = self.pointnet_fp_3((l0_xyz, l2_xyz, None, l2_points))

        concat_feat = tf.concat([up_l4_points, up_l3_points, up_l2_points, l1_points, l0_xyz], axis=-1)
        concat_feat = tf.expand_dims(concat_feat, axis=2)

        new_points_list = []
        for conv1, conv2 in self.up_layers:
            feat = conv1(concat_feat)
            new_points = conv2(feat)
            new_points_list.append(new_points)
        net = tf.concat(new_points_list, axis=1)

        coord = self.conv1(net)
        coord = self.conv2(coord)
        coord = tf.squeeze(coord, axis=[2])

        return coord

    def build_graph(self):
        x = tf.keras.layers.Input((128, 3), batch_size=16)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    @staticmethod
    def get_model(
        input_dims=(16, 128, 3),
        radius=1.0,
        use_batch_normalization=False,
        batch_norm_decay=0.00001,
        up_ratio=4,
        learning_rate=0.001,
    ) -> 'PUNet':
        model = PUNet(
            input_dims=input_dims,
            radius=radius,
            use_batch_normalization=use_batch_normalization,
            batch_norm_decay=batch_norm_decay,
            up_ratio=up_ratio,
        )
        model(np.zeros(input_dims))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=0.9
            ),
            loss=EMD(),
        )

        return model

    def get_config(self):
        config = super().get_config()
        config["input_dims"] = self.input_dims
        config["radius"] = self.radius
        config["use_batch_normalization"] = self.use_batch_normalization
        config["batch_norm_decay"] = self.batch_norm_decay
        config["up_ratio"] = self.up_ratio

        return config
