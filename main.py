import datasets.dyna
import sfpus
import utils
from config import Config

if __name__ == '__main__':
    utils.init_logger()
    Config().print_formatted()
    # datasets.dyna.load_data()

    sfpus.punet.model.get_model(input_shape=(28, 28, 1))

