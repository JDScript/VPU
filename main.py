import utils
from datasets import dyna
from config import Config

if __name__ == '__main__':
    utils.init_logger()
    Config().print_formatted()
    (dataset, steps_per_epoch) = dyna.load_data()
