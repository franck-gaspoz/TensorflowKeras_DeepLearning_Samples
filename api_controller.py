"""
TensorflowKeras_DeepLearning_Samples WEB API
"""
import sys
from typing import Optional
from fastapi import FastAPI
from app import features
from lib import neural_networks
from io import StringIO

# shared vgg16 cnn model
model = ''


def api_init():
    """
    api initialization
    """
    features.initialize()
    print("> api initialized")
    global model
    model = neural_networks.get_vgg16()
    print("> model initialized")


# app api
app = FastAPI()
# shared data
api_init()


@app.get("/classify/best/{image_filename}")
def classify_best(image_filename: str):
    """
    classify the image at the given path
    wget http://127.0.0.1:8000/classify/best/CNN-VGG-mug.jpg
    wget http://127.0.0.1:8000/classify/top-five/talbot-samba-red.jpeg
    :param image_filename: absolute or relative image path
    :return: most probable prediction
    """
    image_filename = "data/" + image_filename
    print(image_filename)
    predict = features.classify_image_using_model_vgg16_cnn(model, image_filename)
    return {
        "type": predict.label[0][0][1],
        "probability": str(predict.label[0][0][2])
    }


@app.get("/classify/top-five/{image_filename}")
def classify_all(image_filename: str):
    """
    classify the image at the given path
    wget http://127.0.0.1:8000/classify/top-five/CNN-VGG-mug.jpg
    wget http://127.0.0.1:8000/classify/top-five/talbot-samba-red.jpeg
    :param image_filename: absolute or relative image path
    :return: most probable prediction
    """
    image_filename = "data/" + image_filename
    print(image_filename)
    predict = features.classify_image_using_model_vgg16_cnn(model, image_filename)

    result = []
    for i in range(len(predict.label[0])):
        print(predict.label[0][i][0])
        result.append({
            "type": predict.label[0][i][1],
            "probability": str(predict.label[0][i][2])
        })

    return result


@app.get("/")
def get_info():
    """
    get info about api
    wget http://127.0.0.1:8000/
    """
    saved_stdout = sys.stdout
    captured_stdout = StringIO()
    sys.stdout = captured_stdout
    model.summary()
    sys.stdout = saved_stdout
    return {
        "api": "TensorflowKeras_DeepLearning_Samples",
        "model": captured_stdout.getvalue()
    }

