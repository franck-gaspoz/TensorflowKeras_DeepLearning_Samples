"""Neural network helper"""


def get_neural_network_config(model):
    """
    Extract info for each layer in a keras model.
    :param model: the model
    :return: a layers description model
    """
    lst_layers = []
    if "Sequential" in str(model):  # -> Sequential doesn't show the input layer
        layer = model.layers[0]
        lst_layers.append({"name": "input",
                           "in": int(layer.input.shape[-1]),
                           "neurons": 0,
                           "out": int(layer.input.shape[-1]),
                           "activation": None,
                           "params": 0,
                           "bias": 0})
    for layer in model.layers:
        try:
            dic_layer = {"name": layer.name,
                         "in": int(layer.input.shape[-1]),
                         "neurons": layer.units,
                         "out": int(layer.output.shape[-1]),
                         "activation": layer.get_config()["activation"],
                         "params": layer.get_weights()[0],
                         "bias": layer.get_weights()[1]}
        except:
            dic_layer = {"name": layer.name,
                         "in": int(layer.input.shape[-1]),
                         "neurons": 0,
                         "out": int(layer.output.shape[-1]),
                         "activation": None,
                         "params": 0,
                         "bias": 0}

        lst_layers.append(dic_layer)
    return lst_layers
