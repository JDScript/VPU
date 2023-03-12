import tensorflow as tf
import pointnet2.layers


def get_model(input_shape):
    # Implement PU-Net using keras functional API
    inputs = tf.keras.layers.Input(shape=input_shape)

    # PointNet Set Abstraction
    h = pointnet2.layers.PointNetSetAbstraction()

    model = tf.keras.models.Model(
        inputs=inputs,
        outputs=h,
    )

    model.summary()

    return model

