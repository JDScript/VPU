import logging
import sys

import h5py
import tensorflow as tf
from os import path

from utils import get_root_path
from config import Config


class _RawDataGenerator:
    def __init__(
        self,
        filepath,
        num_of_patches,
        size_of_dense_patch,
        upsampling_ratio,
    ):
        self.filepath = filepath
        self.num_of_patches = num_of_patches
        self.size_of_dense_patch = size_of_dense_patch
        self.upsampling_ratio = upsampling_ratio
        self.size_of_sparse_patch = size_of_dense_patch // upsampling_ratio

    def get_steps_per_epoch(self):
        with h5py.File(self.filepath, "r") as f:
            dense = f["poisson_1024"][()]
            return len(dense) // Config().DataConfig.batch_size

    @staticmethod
    def output_signature():
        return (
            tf.TensorSpec(
                shape=(256, 3),
                dtype=tf.float64,
            ),
            tf.TensorSpec(
                shape=(1024, 3),
                dtype=tf.float64,
            )
        )

    def __call__(self):
        with h5py.File(self.filepath, "r") as f:
            dense = f["poisson_1024"][()]
            sparse = f["poisson_256"][()]
            for i in range(len(dense)):
                yield (
                    sparse[i],
                    dense[i]
                )


def load_data():
    filepath = path.join(
        get_root_path(),
        Config().DataConfig.pu1k,
    )

    if Config().STAConfig.enable:
        logging.getLogger().error("PU1K dataset can only be used with STA disabled!")
        sys.exit()

    data_generator = _RawDataGenerator(
        filepath=filepath,
        num_of_patches=Config().DataConfig.num_of_patches,
        size_of_dense_patch=Config().DataConfig.size_of_dense_patch,
        upsampling_ratio=Config().SFPUsConfig.upsampling_ratio,
    )

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=data_generator.output_signature(),
    )

    dataset = dataset.batch(Config().DataConfig.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(Config().DataConfig.prefetch)
    dataset = dataset.repeat(10)
    dataset = dataset.shuffle(buffer_size=10000)

    return dataset, data_generator.get_steps_per_epoch()
