"""tensorflow experiments"""

from app import features
from lib import neural_network_view_helper


def main():
    """
    main function
    """
    features.initialize()
    model = features.build_and_show_deep_neural_network()
    neural_network_view_helper.to_image(model)


if __name__ == '__main__':
    main()
