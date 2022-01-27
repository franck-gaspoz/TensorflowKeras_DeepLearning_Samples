"""tensorflow experiments"""

from app import features
from lib import neural_network_view_helper


def main():
    """
    main function
    """
    features.initialize()

    # sample 1 : draw the model

    model = features.build_and_show_deep_neural_network()
    neural_network_view_helper.to_image(model)

    # sample 2 : predict an image with VGG16

    predict = features.classify_image_using_vgg16_cnn("data/CNN-VGG-mug.jpg")
    predict.model.summary()
    print(predict.label)
    print("type: ", predict.label[0][0][1], " with probability: ", predict.label[0][0][2])


if __name__ == '__main__':
    main()
