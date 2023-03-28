import tensorflow as tf
import math
from pointnet2.tf_ops.grouping.tf_grouping import query_ball_point, group_point, knn_point
from pointnet2.tf_ops.approxmatch.tf_approxmatch import approx_match, match_cost
from pointnet2.tf_ops.nn_distance.tf_nndistance import nn_distance
from pointnet2.tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample


@tf.keras.utils.register_keras_serializable()
class Repulsion:
    def __init__(
        self,
        n_sample=20,
        radius=0.07,
        h=0.03,
    ):
        self.n_sample = n_sample
        self.radius = radius
        self.h = h

    def __call__(self, y_true, y_pred):
        idx, points_count = query_ball_point(self.radius, self.n_sample, y_pred, y_pred)
        grouped_pred = group_point(y_pred, idx)
        grouped_pred -= tf.expand_dims(y_pred, idx)

        dist_square = tf.reduce_sum(grouped_pred ** 2, axis=-1)
        dist_square, idx = tf.nn.top_k(-dist_square, 5)
        dist_square = -dist_square[:, :, 1:]
        dist_square = tf.maximum(1e-12, dist_square)
        dist = tf.sqrt(dist_square)
        weight = tf.exp(-dist_square / self.h ** 2)
        uniform_loss = tf.reduce_mean(self.radius - dist * weight)

        return uniform_loss

    def get_config(self):
        return {
            "n_sample": self.n_sample,
            "radius": self.radius,
            "h": self.h,
        }


@tf.keras.utils.register_keras_serializable()
class EMD(tf.keras.losses.Loss):
    def __init__(self, radius=1.0, **kwargs):
        super(EMD, self).__init__(**kwargs)
        self.radius = radius

    def call(self, y_true, y_pred):
        num_of_points = tf.cast(y_true.shape[1], tf.float32)
        match = approx_match(y_pred, y_true)
        cost = match_cost(y_pred, y_true, match)
        cost = cost / self.radius

        return tf.reduce_mean(cost / num_of_points)

    def get_config(self):
        return {
            "radius": self.radius
        }


@tf.keras.utils.register_keras_serializable()
class CD:
    def __init__(
        self,
        radius=1.0,
    ):
        self.radius = radius

    def __call__(self, y_true, y_pred):
        dists_forward, _, dists_backward, _ = nn_distance(y_true, y_pred)
        cd_dist = 0.8 * dists_forward + 0.2 * dists_backward
        cd_dist = tf.reduce_mean(cd_dist, axis=1)
        cd_dist_norm = cd_dist / self.radius

        return tf.reduce_mean(cd_dist_norm)

    def get_config(self):
        return {
            "radius": self.radius
        }


@tf.keras.utils.register_keras_serializable()
class Uniform:
    def __init__(
        self,
        percentages=[0.004, 0.006, 0.008, 0.010, 0.012],
        radius=1.0,
    ):
        self.percentages = percentages
        self.radius = radius

    def __call__(self, y_true, y_pred):
        num_of_points = y_pred.shape[1]
        n_point = int(num_of_points * 0.05)
        loss = []
        for p in self.percentages:
            n_sample = int(num_of_points * p)
            r = math.sqrt(p * self.radius)
            disk_area = math.pi * (self.radius ** 2) * p / n_sample
            new_xyz = gather_point(y_pred, farthest_point_sample(n_point, y_pred))
            idx, pts_cnt = query_ball_point(r, n_sample, y_pred, new_xyz)

            expect_len = tf.sqrt(disk_area)

            grouped_pcd = group_point(y_pred, idx)
            grouped_pcd = tf.concat(tf.unstack(grouped_pcd, axis=1), axis=0)

            var, _ = knn_point(2, grouped_pcd, grouped_pcd)
            uniform_dis = -var[:, :, 1:]
            uniform_dis = tf.sqrt(tf.abs(uniform_dis + 1e-8))
            uniform_dis = tf.reduce_mean(uniform_dis, axis=[-1])
            uniform_dis = tf.square(uniform_dis - expect_len) / (expect_len + 1e-8)
            uniform_dis = tf.reshape(uniform_dis, [-1])

            mean, variance = tf.nn.moments(uniform_dis, axes=0)
            mean = mean * math.pow(p * 100, 2)

            loss.append(mean)

        return tf.add_n(loss) / len(self.percentages)

    def get_config(self):
        return {
            "percentages": self.percentages,
            "radius": self.radius,
        }


@tf.keras.utils.register_keras_serializable()
class Combined:
    def __init__(
        self,
        losses=[],
        losses_weights=[],
        **kwargs,
    ):
        super(Combined, self).__init__(**kwargs)
        self.losses = losses
        self.losses_weights = losses_weights

    def __call__(self, y_true, y_pred):
        combined_loss = tf.constant(0, dtype=tf.float32)
        for loss, weight in zip(self.losses, self.losses_weights):
            combined_loss = tf.add(combined_loss, weight * loss(y_true, y_pred))
        return combined_loss

    def get_config(self):
        return {
            "losses": self.losses,
            "losses_weights": self.losses_weights
        }

