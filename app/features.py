"""
Application features
"""

from lib import tf_cuda
from lib import neural_networks
from lib import neural_network_helper
from lib import neural_network_view_helper
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions


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


def get_image_vgg16_cnn(img_path):
    """
    get and prepare an image for a vgg16 classification
    :param img_path: path of the image
    :return: image
    """
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    return image


def classify_image_using_vgg16_cnn(img_path: str):
    """
    classify an image using a vgg16 model
    :param img_path path of the image to be classified
    :return: ModelVGG16Predict vgg16 model
    """
    model = neural_networks.get_vgg16()
    image = get_image_vgg16_cnn(img_path)
    proba = model.predict(image)
    label = decode_predictions(proba)

    return ModelVGG16Predict(image, model, proba, label)


class ModelVGG16Predict:
    """
    vgg16 cnn image classification inputs and results
    """
    def __init__(self, image, model, proba, label):
        """
        :param image: image to be classified
        :param model: vgg16 cnn model
        :param proba: classification result
        :param label: bests predictions
        """
        self.image = image
        self.model = model
        self.proba = proba
        self.label = label

