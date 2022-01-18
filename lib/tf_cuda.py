"""Tensorflow cuda helper functions"""

import os
import tensorflow as tf
from tensorflow.keras import backend as k


def disable_cuda():
    """
    Disable cuda gpu usage in tensorflow / keras
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


def setup_cuda(cpu_count: int = 1, gpu_count: int = 0, num_cores: int = 4):
    """
    Setup tensorflow/kuda cpu/gpu usage ** doesn't works **
    :param cpu_count: number of cpus
    :param gpu_count: number of gpus
    :param num_cores: number of cores
    """
    config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=num_cores,
        inter_op_parallelism_threads=num_cores,
        allow_soft_placement=True,
        device_count={'CPU': cpu_count,
                      'GPU': gpu_count}
        )
    session = tf.Session(config=config)
    k.set_session(session)


def print_devices_list_no_init():
    """
    print a list of tensorflow physical devices. skip init it
    """
    print('tensorflow devices:')
    print(tf.config.experimental.list_physical_devices())


def print_devices_list():
    """
    list tensorflow physical devices
    """
    print('tensorflow devices:')
    print(tf.config.list_physical_devices())
