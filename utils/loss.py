import tensorflow as tf
import math
from pointnet2.tf_ops.grouping.tf_grouping import query_ball_point, group_point, knn_point, knn_point_2
from pointnet2.tf_ops.approxmatch.tf_approxmatch import approx_match, match_cost
from pointnet2.tf_ops.nn_distance.tf_nndistance import nn_distance
from pointnet2.tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample


@tf.keras.utils.register_keras_serializable()
class Repulsion:
    def __init__(
        self,
        n_sample=20,
        radius=0.07,
        knn=False,
        use_l1=False,
        h=0.001,
    ):
        self.n_sample = n_sample
        self.radius = radius
        self.knn = knn
        self.use_l1 = use_l1
        self.h = h

    def __call__(self, y_true, y_pred):
        if self.knn:
            _, idx = knn_point_2(self.n_sample, y_pred, y_pred)
            pts_cnt = tf.constant(self.n_sample, shape=(30, 1024))
        else:
            idx, pts_cnt = query_ball_point(self.radius, self.n_sample, y_pred, y_pred)

        grouped_pred = group_point(y_pred, idx)  # (batch_size, npoint, nsample, 3)
        grouped_pred -= tf.expand_dims(y_pred, 2)

        # get the uniform loss
        if self.use_l1:
            dists = tf.reduce_sum(tf.abs(grouped_pred), axis=-1)
        else:
            dists = tf.reduce_sum(grouped_pred ** 2, axis=-1)

        val, idx = tf.nn.top_k(-dists, 5)
        val = val[:, :, 1:]  # remove the first one

        h = self.h
        if self.use_l1:
            h = tf.sqrt(self.h) * 2

        val = tf.maximum(0.0, h + val)  # dd/np.sqrt(n)
        repulsion_loss = tf.reduce_mean(val)
        return repulsion_loss

    def get_config(self):
        return {
            "n_sample": self.n_sample,
            "radius": self.radius,
            "knn": self.knn,
            "use_l1": self.use_l1,
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
class HD(tf.keras.losses.Loss):
    def __init__(self, threshold=None):
        super().__init__()
        self.threshold = threshold

    def call(self, y_true, y_pred):
        dists_forward, _, dists_backward, _ = nn_distance(y_true, y_pred)
        if self.threshold is not None:
            forward_threshold = tf.reduce_mean(dists_forward, keepdims=True, axis=1) * self.threshold
            backward_threshold = tf.reduce_mean(dists_backward, keepdims=True, axis=1) * self.threshold
            # only care about distance within threshold (ignore strong outliers)
            dists_forward = tf.where(dists_forward < forward_threshold, dists_forward, tf.zeros_like(dists_forward))
            dists_backward = tf.where(dists_backward < backward_threshold, dists_backward,
                                      tf.zeros_like(dists_backward))
        hd = tf.reduce_max(dists_forward, axis=0)+tf.reduce_max(dists_backward, axis=0)
        return tf.reduce_max(hd)

    def get_config(self):
        return {
            "threshold": self.threshold,
        }



@tf.keras.utils.register_keras_serializable()
class CD:
    def __init__(
        self,
        forward_weight=1.0,
        threshold=None,
    ):
        self.forward_weight = forward_weight
        self.threshold = threshold

    def __call__(self, y_true, y_pred):
        dists_forward, _, dists_backward, _ = nn_distance(y_true, y_pred)
        if self.threshold is not None:
            forward_threshold = tf.reduce_mean(dists_forward, keepdims=True, axis=1) * self.threshold
            backward_threshold = tf.reduce_mean(dists_backward, keepdims=True, axis=1) * self.threshold
            # only care about distance within threshold (ignore strong outliers)
            dists_forward = tf.where(dists_forward < forward_threshold, dists_forward, tf.zeros_like(dists_forward))
            dists_backward = tf.where(dists_backward < backward_threshold, dists_backward,
                                      tf.zeros_like(dists_backward))
        dists_forward = tf.reduce_mean(dists_forward, axis=1)
        dists_backward = tf.reduce_mean(dists_backward, axis=1)
        cd_dist = self.forward_weight * dists_forward + dists_backward
        cd_loss = tf.reduce_mean(cd_dist)

        return cd_loss

    def get_config(self):
        return {
            "forward_weight": self.forward_weight,
            "threshold": self.threshold,
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

