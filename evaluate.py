import sta.module
import sfpus.mpu.model
import utils
from datasets import dyna
from config import Config
import tensorflow as tf
import open3d as o3d
from utils.loss import EMD
from utils.loss import CD
from utils.loss import HD

if __name__ == '__main__':
    utils.init_logger()
    Config().print_formatted()
    _, test_dataset = dyna.load_data()

    # module = sta.module.STAModule()
    model = sfpus.mpu.model.MPU.get_model(
        input_dims=(16, 128, 3),
        # input_dims=(16, 3, 128, 3),
        # sta_module=module
    )

    model.summary()

    model.load_weights("./saved_models/MPU_non_sta.h5")
    # model.load_weights("./saved_models/MPU_sta.h5")

    emd_class = EMD()
    cd_class = CD()
    hd_class = HD()

    emd = []
    cd = []
    hd = []
    i = 0
    for sparse, gt in test_dataset:
        prediction = model.predict_on_batch(sparse)
        gt = tf.cast(gt, dtype=tf.float32)
        emd.append(emd_class(gt, prediction))
        cd.append(cd_class(gt, prediction))
        hd.append(hd_class(gt, prediction))
        i += 1
        print(f"\rPredicted: {i}", end="")

    print()
    print(tf.reduce_mean(emd))
    print(tf.reduce_mean(cd))
    print(tf.reduce_mean(hd))

    # i = test_dataset.as_numpy_iterator()
    # d = next(i)  # (16, 3, 128, 3), (16, 512, 3)
    #
    # sparse_point_clouds, dense_point_clouds = d
    # predictions = model.predict_on_batch(sparse_point_clouds)
    #
    # # sparse = o3d.geometry.PointCloud()
    # # sparse.points = o3d.utility.Vector3dVector(sparse_point_clouds[:, :, :].reshape(2048, 3).tolist())
    #
    # dense = o3d.geometry.PointCloud()
    # dense.points = o3d.utility.Vector3dVector(dense_point_clouds.reshape(8192, 3).tolist())
    #
    # prediction = o3d.geometry.PointCloud()
    # prediction.points = o3d.utility.Vector3dVector(predictions.reshape(8192, 3).tolist())
    # print(predictions.shape)
    # print(predictions.reshape(8192, 3).shape)
    #
    # # o3d.visualization.draw_geometries([sparse])
    # o3d.visualization.draw_geometries([dense])
    # o3d.visualization.draw_geometries([prediction])
    #
    # length = len()
    # for gt, input in dense_point_clouds, predictions:
    #
    # emd = EMD()
    # print(emd(dense_point_clouds, predictions))
    #
    # cd = CD()
    # print(cd(dense_point_clouds, predictions))
    #
    # hd = HD()
    # print(hd(dense_point_clouds, predictions))

