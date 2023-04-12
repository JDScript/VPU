import logging
import os.path
import sys

import yaml
from enum import Enum

from utils import Singleton, get_root_path


class SFPUs(Enum):
    PU_NET = "PU-Net"
    PU_GCN = "PU-GCN"
    MPU = "MPU"
    PU_GAN = "PU-GAN"


class _STAConfig:
    enable = True
    frames = 3

    def __init__(self, cfg_dict=None):
        if not isinstance(cfg_dict, dict):
            return
        if cfg_dict.get("enable") is not None:
            self.enable = cfg_dict.get("enable")
        if cfg_dict.get("frames") is not None:
            self.frames = cfg_dict.get("frames")

    def to_dict(self):
        return {
            "enable": self.enable,
            "frames": self.frames,
        }


class _DataConfig:
    dyna = "data/dyna_poisson_8192_resampled.h5"
    pu1k = "data/pu1k.h5"
    num_of_patches = 16
    size_of_dense_patch = 512
    batch_size = 16
    prefetch = 10
    window_stride = 1
    train_split_ratio = 0.8

    def __init__(self, cfg_dict=None):
        if not isinstance(cfg_dict, dict):
            return
        if cfg_dict.get("dyna") is not None:
            self.dyna = cfg_dict.get("dyna")
        if cfg_dict.get("pu1k") is not None:
            self.pu1k = cfg_dict.get("pu1k")
        if cfg_dict.get("num_of_patches") is not None:
            self.num_of_patches = cfg_dict.get("num_of_patches")
        if cfg_dict.get("size_of_dense_patch") is not None:
            self.size_of_dense_patch = cfg_dict.get("size_of_dense_patch")
        if cfg_dict.get("batch_size") is not None:
            self.batch_size = cfg_dict.get("batch_size")
        if cfg_dict.get("prefetch") is not None:
            self.prefetch = cfg_dict.get("prefetch")
        if cfg_dict.get("window_stride") is not None:
            self.window_stride = cfg_dict.get("window_stride")
        if cfg_dict.get("train_split_ratio") is not None:
            self.train_split_ratio = cfg_dict.get("train_split_ratio")

    def to_dict(self):
        return {
            "dyna": self.dyna,
            "pu1k": self.pu1k,
            "num_of_patches": self.num_of_patches,
            "size_of_dense_patch": self.size_of_dense_patch,
            "batch_size": self.batch_size,
            "prefetch": self.prefetch,
            "window_stride": self.window_stride,
            "train_split_ratio": self.train_split_ratio,
        }


class _SFPUsConfig:
    use = SFPUs.MPU
    upsampling_ratio = 4
    learning_rate = 0.001

    def __init__(self, cfg_dict=None):
        if not isinstance(cfg_dict, dict):
            return

        if cfg_dict.get("use") is not None:
            try:
                use = SFPUs(cfg_dict.get("use"))
                self.use = use
            except ValueError as e:
                logging.getLogger().error(f"Config Error: {e}, in (\"PU-Net\", \"PU-GCN\", \"MPU\", \"PU-GAN\")")
                sys.exit()
        if cfg_dict.get("upsampling_ratio"):
            self.upsampling_ratio = cfg_dict.get("upsampling_ratio")
        if cfg_dict.get("learning_rate"):
            self.learning_rate = cfg_dict.get("learning_rate")

    def to_dict(self):
        return {
            "use": self.use.value,
            "upsampling_ratio": self.upsampling_ratio,
            "learning_rate": self.learning_rate,
        }


class Config(metaclass=Singleton):
    def __init__(self):
        # Read configurations from config.yaml
        cfg = dict()

        try:
            stream = open(os.path.join(get_root_path(), "config.yaml"), "r")
            cfg = yaml.safe_load(stream)
        except FileNotFoundError:
            logging.getLogger().info(f"Error loading config, using default config instead")
        except yaml.YAMLError:
            logging.getLogger().info(f"Error parsing config, using default config instead")

        self.STAConfig = _STAConfig(cfg.get("STA"))
        self.SFPUsConfig = _SFPUsConfig(cfg.get("SFPUs"))
        self.DataConfig = _DataConfig(cfg.get("data"))
        return

    def print_formatted(self):
        # Print formatted configuration file
        cfg_dict = {
            "VPU Framework Config": {
                "STA": self.STAConfig.to_dict(),
                "SFPUs": self.SFPUsConfig.to_dict(),
                "data": self.DataConfig.to_dict(),
            }
        }

        logging.getLogger().info(yaml.dump(cfg_dict, sort_keys=False, default_flow_style=False))
