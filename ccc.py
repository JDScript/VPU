import utils
from datasets import dyna, pu1k
from config import Config
from sfpus.mpu.model import MPU
import tensorflow as tf
import open3d as o3d
import numpy as np

if __name__ == '__main__':
    utils.init_logger()
    Config().print_formatted()
    # (dataset, steps_per_epoch) = pu1k.load_data()
    # (dataset, steps_per_epoch) = dyna.load_data()
    model = MPU.get_model(
        input_dims=(
            Config().DataConfig.batch_size,
            Config().DataConfig.size_of_dense_patch // Config().SFPUsConfig.upsampling_ratio,
            3
        ),
        up_ratio=Config().SFPUsConfig.upsampling_ratio
    )

    model.summary()
    #
    # model.fit(
    #     dataset,
    #     steps_per_epoch=steps_per_epoch,
    #     epochs=10,
    #     callbacks=[
    #         tf.keras.callbacks.ModelCheckpoint(
    #             filepath="./saved_models/MPU.h5",
    #             save_weights_only=True,
    #             monitor="loss",
    #             save_best_only=False,
    #             verbose=False,
    #         ),
    #     ]
    # )
    #
    # model.load_weights("./saved_models/MPU.h5")
    # model.summary()
    #
    # i = dataset.as_numpy_iterator()
    # d = next(i)
    #
    # sparse_point_clouds, dense_point_clouds = d
    # predictions = model.predict_on_batch(sparse_point_clouds)
    #
    # print(sparse_point_clouds[0])
    #
    # sparse = o3d.geometry.PointCloud()
    # sparse.points = o3d.utility.Vector3dVector(sparse_point_clouds[0].tolist())
    #
    # dense = o3d.geometry.PointCloud()
    # dense.points = o3d.utility.Vector3dVector(dense_point_clouds[0].tolist())
    #
    # prediction = o3d.geometry.PointCloud()
    # prediction.points = o3d.utility.Vector3dVector(predictions[0].tolist())
    #
    # o3d.visualization.draw_geometries([sparse])
    # o3d.visualization.draw_geometries([dense])
    # o3d.visualization.draw_geometries([prediction])
