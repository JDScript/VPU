import sta.module
import sfpus.mpu.model
import utils
from datasets import dyna
from config import Config
import tensorflow as tf
import open3d as o3d

if __name__ == '__main__':
    utils.init_logger()
    Config().print_formatted()
    train_dataset, _ = dyna.load_data()

    # module = sta.module.STAModule()
    model = sfpus.mpu.model.MPU.get_model(
        input_dims=(16, 128, 3),
        # sta_module=module
    )

    model.summary()

    # model.load_weights("./saved_models/MPU.h5")

    model.fit(
        train_dataset,
        epochs=10,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath="./saved_models/MPU_non_sta.h5",
                save_weights_only=True,
                monitor="loss",
                save_best_only=False,
                verbose=1
            ),
        ]
    )

    # i = dataset.as_numpy_iterator()
    # d = next(i) # (16, 3, 128, 3), (16, 512, 3)
    # for _ in range(10000):
    #     d = next(i)
    #
    # sparse_point_clouds, dense_point_clouds = d
    # predictions = model.predict_on_batch(sparse_point_clouds)
    #
    # sparse = o3d.geometry.PointCloud()
    # sparse.points = o3d.utility.Vector3dVector(sparse_point_clouds[:, 1, :, :].reshape(2048, 3).tolist())
    #
    # dense = o3d.geometry.PointCloud()
    # dense.points = o3d.utility.Vector3dVector(dense_point_clouds.reshape(8192, 3).tolist())
    #
    # prediction = o3d.geometry.PointCloud()
    # prediction.points = o3d.utility.Vector3dVector(predictions.reshape(8192, 3).tolist())
    # print(predictions.shape)
    # print(predictions.reshape(8192, 3).shape)
    #
    # o3d.visualization.draw_geometries([sparse])
    # o3d.visualization.draw_geometries([dense])
    # o3d.visualization.draw_geometries([prediction])

