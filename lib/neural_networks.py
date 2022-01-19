"""Neural networks samples"""

from tensorflow.keras import models, layers
from keras.applications.vgg16 import VGG16
import os.path


def get_vgg16():
    """
    get a vgg16 from pre-trained production model
    stores it locally in current path (data/vgg16.h5)
    keras already cache it in ~/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5
    :return: model
    """
    model = VGG16()
    # save a local version of the model, do not overwrite
    path = "data/vgg16.h5"
    if not os.path.isfile(path):
        model.save(path)
    return model


def get_fully_connected_layer():
    """
    Single fully connected layer 3x1
    :return: model
    """
    model = models.Sequential(name="Perceptron", layers=[
        layers.Dense(  # a fully connected layer
            name="dense",
            input_dim=3,  # with 3 features as the input
            units=1,  # and 1 node because we want 1 output
            activation='linear'  # f(x)=x
        )
    ])
    return model


def get_deep_neural_network(nb_features: int = 10):
    """
    2 hidden layers, 1 output
    :param nb_features: number of inputs (default 10)
    :return: model
    """
    model = models.Sequential(name="DeepNN", layers=[
        # hidden layer 1
        layers.Dense(name="h1", input_dim=nb_features,
                     units=int(round((nb_features + 1) / 2)),
                     activation='relu'),
        layers.Dropout(name="drop1", rate=0.2),

        # hidden layer 2
        layers.Dense(name="h2", units=int(round((nb_features + 1) / 4)),
                     activation='relu'),
        layers.Dropout(name="drop2", rate=0.2),

        # layer output
        layers.Dense(name="output", units=1, activation='sigmoid')
    ])
    return model


def get_deep_neural_network_perceptron(nb_features: int = 10):
    """
    1 input, 2 hidden layers, output, sigmoid activation
    :param nb_features: number of inputs (default 10)
    :return: model
    """
    # Perceptron
    inputs = layers.Input(name="input", shape=(3,))
    outputs = layers.Dense(name="output", units=1,
                           activation='linear')(inputs)
    model = models.Model(inputs=inputs, outputs=outputs,
                         name="Perceptron")

    # DeepNN
    # layer input
    inputs = layers.Input(name="input", shape=(nb_features,))
    # hidden layer 1
    h1 = layers.Dense(name="h1", units=int(round((nb_features + 1) / 2)), activation='relu')(inputs)
    h1 = layers.Dropout(name="drop1", rate=0.2)(h1)
    # hidden layer 2
    h2 = layers.Dense(name="h2", units=int(round((nb_features + 1) / 4)), activation='relu')(h1)
    h2 = layers.Dropout(name="drop2", rate=0.2)(h2)
    # layer output
    outputs = layers.Dense(name="output", units=1, activation='sigmoid')(h2)
    model = models.Model(inputs=inputs, outputs=outputs, name="DeepNN")
    return model

