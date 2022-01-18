"""
Application features
"""

from lib import tf_cuda
from lib import neural_networks
from lib import neural_network_helper
from lib import neural_network_view_helper


def initialize():
    """
    Initialize application
    """
    tf_cuda.disable_cuda()
    tf_cuda.print_devices_list_no_init()


def show_model(model):
    """
    Show a model and print informations about it
    :param model: the model
    :return: model
    """
    model.summary()
    layers_config = neural_network_helper.get_neural_network_config(model)
    neural_network_view_helper.visualize_nn(model, layers_config)
    return model


def build_and_show_fully_connected_layer():
    """
    Build and show a single fully connected layer
    :return: model
    """
    model = neural_networks.get_fully_connected_layer()
    show_model(model)
    return model


def build_and_show_deep_neural_network():
    """
    Build and show a deep neural network
    :return: model
    """
    model = neural_networks.get_deep_neural_network_perceptron()
    show_model(model)
    return model

