import h5py
from os import path

import pointnet2.tf_ops.grouping.tf_grouping
from utils import get_root_path
from config import Config

def load_data():
    filepath = path.join(
        get_root_path(),
        Config().DataSourcesConfig.dyna,
    )

    # Load file from current directory
    file = h5py.File(filepath, "r")