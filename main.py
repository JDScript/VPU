import sta.module
import sfpus.mpu.model
import utils
from datasets import dyna
from config import Config
import tensorflow as tf
from datetime import datetime

if __name__ == '__main__':
    utils.init_logger()
    Config().print_formatted()
    (dataset, steps_per_epoch) = dyna.load_data()

    module = sta.module.STAModule()
    model = sfpus.mpu.model.MPU.get_model(
        # input_dims=(16, 3, 128, 3),
        # sta_module=module
    )

    model.summary()

    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
                                                     histogram_freq=1,
                                                     profile_batch='500,520')

    model.fit(
        dataset,
        steps_per_epoch=steps_per_epoch - 43000,
        epochs=10,
        callbacks=[
            tboard_callback,
            tf.keras.callbacks.ModelCheckpoint(
                filepath="./saved_models/MPU.h5",
                save_weights_only=True,
                monitor="loss",
                save_best_only=False,
                verbose=False,
            ),
        ]
    )