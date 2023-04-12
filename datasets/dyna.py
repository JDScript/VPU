from os import path
from utils import get_root_path
from config import Config
from sklearn.model_selection import train_test_split
from pointnet2.tf_ops.sampling.tf_sampling import farthest_point_sample
from pointnet2.tf_ops.grouping.tf_grouping import knn_point
import tensorflow as tf
import numpy as np
import h5py


class _RawDataGenerator:
    def __init__(
        self,
        filepath,
        frames,
        num_of_patches,
        size_of_dense_patch,
        upsampling_ratio,
        enable_sta,
        window_stride=1,
        sequences=[],
    ):
        self.filepath = filepath
        self.frames = frames
        self.num_of_patches = num_of_patches
        self.size_of_dense_patch = size_of_dense_patch
        self.upsampling_ratio = upsampling_ratio
        self.size_of_sparse_patch = size_of_dense_patch // upsampling_ratio
        self.enable_sta = enable_sta
        self.window_stride = window_stride
        self.sequences = sequences

    def output_signature(self):
        if self.enable_sta:
            return (
                tf.TensorSpec(
                    shape=(self.frames, self.size_of_sparse_patch, 3),
                    dtype=tf.float64,
                ),
                tf.TensorSpec(
                    shape=(self.size_of_dense_patch, 3),
                    dtype=tf.float64,
                )
            )
        return (
            tf.TensorSpec(
                shape=(self.size_of_sparse_patch, 3),
                dtype=tf.float64,
            ),
            tf.TensorSpec(
                shape=(self.size_of_dense_patch, 3),
                dtype=tf.float64,
            )
        )

    def _gen_multi_frames(self):
        with h5py.File(self.filepath) as f:
            for sequence_key in self.sequences:
                sequence = f[sequence_key][()]
                seeds_indices = farthest_point_sample(
                    self.num_of_patches,
                    sequence[self.frames // 2: len(sequence) - self.frames // 2: self.window_stride]
                )
                for idx, seeds_index in enumerate(seeds_indices):
                    mid_frame_idx = self.frames // 2 + idx * self.window_stride
                    seeds = sequence[mid_frame_idx][seeds_index]
                    frames_group = sequence[mid_frame_idx - self.frames // 2: mid_frame_idx + self.frames // 2 + 1]
                    _, patch_indices = knn_point(
                        self.size_of_dense_patch,
                        frames_group,
                        np.tile(seeds, (self.frames, 1, 1))
                    )

                    dense_patches = np.array([
                        frames_group[i][patch_indices[i]]
                        for i in range(self.frames)
                    ])

                    ground_truth = dense_patches[self.frames // 2]

                    sparse_patches = np.transpose(
                        dense_patches[:, :, np.random.choice(
                            self.size_of_dense_patch,
                            self.size_of_sparse_patch,
                            False
                        )],
                        (1, 0, 2, 3)
                    )

                    yield from zip(sparse_patches, ground_truth)

    def _gen_single_frame(self):
        with h5py.File(self.filepath, "r") as f:
            for sequence_key in self.sequences:
                # Using window stride to skip some frames
                sequence = f[sequence_key][()][::self.window_stride]  # (None, 8192, 3)
                seeds_indices = farthest_point_sample(
                    self.num_of_patches,
                    sequence,
                )  # (None, num_of_patches)
                for idx, seeds_index in enumerate(seeds_indices):
                    seeds = sequence[idx][seeds_index]
                    _, patch_indices = knn_point(
                        self.size_of_dense_patch,
                        np.expand_dims(sequence[idx], axis=0),
                        np.expand_dims(seeds, axis=0),
                    )  # (1, num_of_seeds, size_of_dense_patch)
                    patch_indices = patch_indices[0]  # (num_of_seeds, size_of_dense_patch)
                    dense_patches = sequence[idx][patch_indices]  # (num_of_seeds, size_of_dense_patch, 3)
                    sparse_indices = np.random.permutation(512)[:128]
                    sparse_patches = dense_patches[:, sparse_indices, :]
                    yield from zip(sparse_patches, dense_patches)

    def __call__(self, *args, **kwargs):
        if self.enable_sta:
            return self._gen_multi_frames()
        else:
            return self._gen_single_frame()


def load_data() -> (tf.data.Dataset, tf.data.Dataset):
    filepath = path.join(
        get_root_path(),
        Config().DataConfig.dyna,
    )

    # Split data to train and test set
    with h5py.File(filepath) as f:
        train_keys, test_keys = train_test_split(
            list(f.keys()),
            train_size=Config().DataConfig.train_split_ratio,
            shuffle=False
        )

    train_data_generator = _RawDataGenerator(
        filepath=filepath,
        frames=Config().STAConfig.frames,
        num_of_patches=Config().DataConfig.num_of_patches,
        size_of_dense_patch=Config().DataConfig.size_of_dense_patch,
        upsampling_ratio=Config().SFPUsConfig.upsampling_ratio,
        enable_sta=Config().STAConfig.enable,
        window_stride=Config().DataConfig.window_stride,
        sequences=train_keys,
    )

    test_data_generator = _RawDataGenerator(
        filepath=filepath,
        frames=Config().STAConfig.frames,
        num_of_patches=Config().DataConfig.num_of_patches,
        size_of_dense_patch=Config().DataConfig.size_of_dense_patch,
        upsampling_ratio=Config().SFPUsConfig.upsampling_ratio,
        enable_sta=Config().STAConfig.enable,
        window_stride=Config().DataConfig.window_stride,
        sequences=test_keys,
    )

    train_dataset = tf.data.Dataset.from_generator(
        train_data_generator,
        output_signature=train_data_generator.output_signature()
    )

    test_dataset = tf.data.Dataset.from_generator(
        test_data_generator,
        output_signature=test_data_generator.output_signature()
    )

    train_dataset = train_dataset.batch(Config().DataConfig.batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(Config().DataConfig.prefetch)
    train_dataset = train_dataset.cache()

    test_dataset = test_dataset.batch(Config().DataConfig.batch_size, drop_remainder=True)
    test_dataset = test_dataset.prefetch(Config().DataConfig.prefetch)
    test_dataset = test_dataset.cache()

    return train_dataset, test_dataset
