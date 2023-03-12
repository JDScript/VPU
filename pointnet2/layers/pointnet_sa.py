import tensorflow as tf
import numpy as np
from ..tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point
from ..tf_ops.grouping.tf_grouping import query_ball_point, group_point, knn_point
from ..tf_ops.interpolation.tf_interpolate import three_nn, three_interpolate
from pointnet_conv2d import PointNetConv2D


class PointNetSetAbstraction(tf.keras.layers.Layer):
    """
    PointNet Set Abstraction Module implemented as tf.keras.layers.Layer

    It is not trainable, so won't implement build method.
    """

    def __init__(
        self,
        num_of_fps_points: int,
        radius: float,
        num_of_local_points: int,
        mlp_output_shape_each_point: list[int],
        mlp_output_shape_each_region: list[int] | None = None,
        group_all=False,
        use_batch_normalization=True,
        batch_norm_decay=0.00001,
        pooling="max",
        knn=False,
        use_xyz=True,
    ):
        self.num_of_fps_points = num_of_fps_points
        self.radius = radius
        self.num_of_local_points = num_of_local_points
        self.mlp_output_shape_each_point = mlp_output_shape_each_point
        self.mlp_output_shape_each_region = mlp_output_shape_each_region
        self.group_all = group_all
        self.use_batch_normalization = use_batch_normalization
        self.batch_norm_decay = batch_norm_decay
        self.pooling = pooling
        self.knn = knn
        self.use_xyz = use_xyz
        return

    def call(self, xyz, points, training=True):

        if self.group_all:
            self.num_of_local_points = xyz.shape[1]
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz=self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(
                npoint=self.num_of_fps_points,
                radius=self.radius,
                nsample=self.num_of_local_points,
                xyz=xyz,
                points=points,
                knn=self.knn,
                use_xyz=self.use_xyz,
            )

        if self.mlp_output_shape_each_region is None:
            self.mlp_output_shape_each_region = []

        # Point Feature Embedding
        for i, num_out_channel in enumerate(self.mlp_output_shape_each_point):
            new_points = PointNetConv2D(
                num_out_channel,
                (1, 1),
                padding="VALID",
                use_batch_normalization=self.use_batch_normalization,
                batch_norm_decay=self.batch_norm_decay,
                strides=(1, 1)
            )(new_points)

        # Pooling in Local Regions
        if self.pooling == "max":
            new_points = tf.reduce_max(new_points, axis=[2], keepdims=True)
        elif self.pooling == "avg":
            new_points = tf.reduce_mean(new_points, axis=[2], keepdims=True)
        elif self.pooling == "max_and_avg":
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True)
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True)
            new_points = tf.concat([avg_points, max_points], axis=-1)
        elif self.pooling == "weighted_avg":
            dists = tf.norm(grouped_xyz, axis=-1, ord=2, keepdims=True)
            exp_dists = tf.exp(-dists * 5)
            weights = exp_dists / tf.reduce_sum(exp_dists, axis=2, keepdims=True)  # (batch_size, npoint, nsample, 1)
            new_points *= weights  # (batch_size, npoint, nsample, mlp[-1])
            new_points = tf.reduce_sum(new_points, axis=2, keepdims=True)

        if self.mlp_output_shape_each_region is not None:
            for i, num_out_channel in enumerate(self.mlp_output_shape_each_region):
                new_points = PointNetConv2D(
                    num_out_channel,
                    (1, 1),
                    padding='VALID',
                    use_batch_normalization=self.use_batch_normalization,
                    batch_norm_decay=self.batch_norm_decay,
                    strides=(1, 1)
                )(new_points)

        new_points = tf.squeeze(new_points, axis=[2])
        return new_xyz, new_points, idx


# Method copied from PointNet++
def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):
    """
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    """

    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))  # (batch_size, npoint, 3)
    if knn:
        _, idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])  # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


# Method copied from PointNet++
def sample_and_group_all(xyz, points, use_xyz=True):
    """
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    """
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0, 0, 0]).reshape((1, 1, 3)), (batch_size, 1, 1)),
                          dtype=tf.float32)  # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1, 1, nsample)), (batch_size, 1, 1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3))  # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2)  # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1)  # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz
