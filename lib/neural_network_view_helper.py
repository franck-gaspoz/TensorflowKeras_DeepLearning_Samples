"""Neural network visualization helper functions"""

import matplotlib.pyplot as plt
from tensorflow.keras import utils


def to_image(model,
             to_file: str = 'data/model.png',
             show_shapes: bool = True,
             show_dtypes: bool = True,
             show_layer_names: bool = True,
             rank_dir: str = 'TB',
             expand_nested: bool = False,
             dpi: int = 96):
    """
    get a model image
    :param model: a keras model
    :param to_file: output file (default 'model.png')
    :param show_shapes: weather or not show shapes (default True)
    :param show_dtypes: weather or not show dtypes (default True)
    :param show_layer_names: weather or not show names (default True)
    :param rank_dir: 'TB' creates a vertical plot; 'LR' creates a horizontal plot (default 'TB')
    :param expand_nested: Whether to expand nested models into clusters (default False)
    :param dpi: dots per inch (default 96)
    """
    utils.plot_model(model, to_file, show_shapes, show_dtypes, show_layer_names, rank_dir, expand_nested, dpi)


def visualize_nn(model, layers_config, description: bool = True, figure_size=(10, 8)):
    """
    Plot the structure of a keras neural network.
    :param model: a keras model
    :param layers_config: neural_network_helper.get_neural_network_config
    :param description:  hide/show description
    :param figure_size: (x,y) - figure size
    """
    # get layers info
    # lst_layers = utils_nn_config(model)
    lst_layers = layers_config

    layer_sizes = [layer["out"] for layer in lst_layers]

    # fig setup
    fig = plt.figure(figsize=figure_size)
    ax = fig.gca()
    ax.set(title=model.name)
    ax.axis('off')
    left, right, bottom, top = 0.1, 0.9, 0.1, 0.9
    x_space = (right - left) / float(len(layer_sizes) - 1)
    y_space = (top - bottom) / float(max(layer_sizes))
    p = 0.025

    # nodes
    for i, n in enumerate(layer_sizes):
        top_on_layer = y_space * (n - 1) / 2.0 + (top + bottom) / 2.0
        layer = lst_layers[i]
        color = "green" if i in [0, len(layer_sizes) - 1] else "blue"
        color = "red" if (layer['neurons'] == 0) and (i > 0) else color

        # add description
        if description is True:
            d = i if i == 0 else i - 0.5
            if layer['activation'] is None:
                plt.text(x=left + d * x_space, y=top, fontsize=10, color=color, s=layer["name"].upper())
            else:
                plt.text(x=left + d * x_space, y=top, fontsize=10, color=color, s=layer["name"].upper())
                plt.text(x=left + d * x_space, y=top - p, fontsize=10, color=color, s=layer['activation'] + " (")
                plt.text(x=left + d * x_space, y=top - 2 * p, fontsize=10, color=color,
                         s="Σ" + str(layer['in']) + "[X*w]+b")
                out = " Y" if i == len(layer_sizes) - 1 else " out"
                plt.text(x=left + d * x_space, y=top - 3 * p, fontsize=10, color=color,
                         s=") = " + str(layer['neurons']) + out)

        # circles
        for m in range(n):
            color = "limegreen" if color == "green" else color
            circle = plt.Circle(xy=(left + i * x_space, top_on_layer - m * y_space - 4 * p), radius=y_space / 4.0,
                                color=color, ec='k', zorder=4)
            ax.add_artist(circle)

            # add text
            if i == 0:
                plt.text(x=left - 4 * p, y=top_on_layer - m * y_space - 4 * p, fontsize=10,
                         s=r'$X_{' + str(m + 1) + '}$')
            elif i == len(layer_sizes) - 1:
                plt.text(x=right + 4 * p, y=top_on_layer - m * y_space - 4 * p, fontsize=10,
                         s=r'$y_{' + str(m + 1) + '}$')
            else:
                plt.text(x=left + i * x_space + p,
                         y=top_on_layer - m * y_space + (y_space / 8. + 0.01 * y_space) - 4 * p, fontsize=10,
                         s=r'$H_{' + str(m + 1) + '}$')

    # links
    for i, (n_a, n_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer = lst_layers[i + 1]
        color = "green" if i == len(layer_sizes) - 2 else "blue"
        color = "red" if layer['neurons'] == 0 else color
        layer_top_a = y_space * (n_a - 1) / 2. + (top + bottom) / 2. - 4 * p
        layer_top_b = y_space * (n_b - 1) / 2. + (top + bottom) / 2. - 4 * p
        for m in range(n_a):
            for o in range(n_b):
                line = plt.Line2D([i * x_space + left, (i + 1) * x_space + left],
                                  [layer_top_a - m * y_space, layer_top_b - o * y_space],
                                  c=color, alpha=0.5)
                if layer['activation'] is None:
                    if o == m:
                        ax.add_artist(line)
                else:
                    ax.add_artist(line)
    plt.show()
