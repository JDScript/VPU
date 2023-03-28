import h5py
import tensorflow as tf
import numpy as np
from os import path

from pointnet2.tf_ops.sampling.tf_sampling import farthest_point_sample
from pointnet2.tf_ops.grouping.tf_grouping import knn_point
from utils import get_root_path
from config import Config


class _RawDataGenerator:
    def __init__(
        self,
        filepath,
        frames,
        num_of_patches,
        size_of_dense_patch,
        upsampling_ratio,
        enable_sta,
        enable_patch,
    ):
        self.filepath = filepath
        self.frames = frames
        self.num_of_patches = num_of_patches
        self.size_of_dense_patch = size_of_dense_patch
        self.upsampling_ratio = upsampling_ratio
        self.size_of_sparse_patch = size_of_dense_patch // upsampling_ratio
        self.enable_sta = enable_sta
        self.enable_patch = enable_patch

    def get_steps_per_epoch(self):
        with h5py.File(self.filepath, "r") as f:
            frames = 0
            for sequence_name in f.keys():
                frames += len(f[sequence_name][()])
            return frames * self.num_of_patches // Config().DataConfig.batch_size

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

    def _multi_frame_gen(self):
        with h5py.File(self.filepath, "r") as f:
            for sequence_name in f.keys():
                sequence = f[sequence_name][()]
                # Using sliding window to load frames as an array
                num_of_frame_groups = len(sequence) - self.frames + 1
                all_seeds_indices = farthest_point_sample(
                    self.num_of_patches,
                    sequence[self.frames // 2: len(sequence) - self.frames // 2]
                )

                for frame_index in range(0, num_of_frame_groups):
                    # Cut into patches, using fps and knn
                    mid_frame_idx = frame_index + self.frames // 2
                    mid_frame = sequence[mid_frame_idx]
                    # FPS from middle(ground truth) frame
                    seeds = mid_frame[all_seeds_indices[frame_index]]
                    # Copying seeds from ground truth frame along first axis
                    seeds_for_frames = np.tile(seeds, (self.frames, 1, 1))

                    _, patch_indices = knn_point(
                        self.size_of_dense_patch,
                        sequence[frame_index: frame_index + self.frames],
                        seeds_for_frames,
                    )

                    for patch_idx in range(self.num_of_patches):
                        dense_patches = np.array([
                            sequence[frame_index + frame][patch_indices[frame][patch_idx]]
                            for frame in range(self.frames)
                        ])

                        ground_truth = dense_patches[self.frames // 2]

                        sparse_patches = np.array([
                            p[np.random.choice(
                                self.size_of_dense_patch,
                                self.size_of_sparse_patch,
                                False
                            )]
                            for p in dense_patches
                        ])

                        yield sparse_patches, ground_truth

    def _single_frame_gen(self):
        with h5py.File(self.filepath, "r") as f:
            for sequence_name in f.keys():
                sequence = f[sequence_name][()]

                if not self.enable_patch:
                    for frame_idx in range(len(sequence)):
                        frame = sequence[frame_idx]
                        dense_frame = frame[np.random.choice(
                            8192,
                            self.size_of_dense_patch,
                            False
                        )]
                        sparse_frame = dense_frame[np.random.choice(
                            self.size_of_dense_patch,
                            self.size_of_sparse_patch,
                            False
                        )]
                        yield sparse_frame, dense_frame
                else:
                    all_seeds_indices = farthest_point_sample(
                        self.num_of_patches,
                        sequence
                    )

                    for frame_idx in range(len(sequence)):
                        frame = sequence[frame_idx]

                        _, patch_indices = knn_point(
                            self.size_of_dense_patch,
                            np.expand_dims(frame, axis=0),
                            np.expand_dims(frame[all_seeds_indices[frame_idx]], axis=0),
                        )

                        patch_indices = patch_indices[0]

                        for patch_idx in range(self.num_of_patches):
                            dense_patch = frame[patch_indices[patch_idx]]
                            sparse_patch = dense_patch[np.random.choice(
                                self.size_of_dense_patch,
                                self.size_of_sparse_patch,
                                False
                            )]
                            yield sparse_patch, dense_patch

    def __call__(self):
        if self.enable_sta:
            return self._multi_frame_gen()
        else:
            return self._single_frame_gen()


def load_data() -> (tf.data.Dataset, int):
    filepath = path.join(
        get_root_path(),
        Config().DataConfig.dyna,
    )

    # Load file from current directory
    data_generator = _RawDataGenerator(
        filepath=filepath,
        frames=Config().STAConfig.frames,
        num_of_patches=Config().DataConfig.num_of_patches,
        size_of_dense_patch=Config().DataConfig.size_of_dense_patch,
        upsampling_ratio=Config().SFPUsConfig.upsampling_ratio,
        enable_sta=Config().STAConfig.enable,
        enable_patch=Config().DataConfig.enable_patch,
    )

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=data_generator.output_signature()
    )

    dataset = dataset.batch(Config().DataConfig.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(Config().DataConfig.prefetch)
    dataset = dataset.cache()
    dataset = dataset.repeat(10)

    return dataset, data_generator.get_steps_per_epoch()
