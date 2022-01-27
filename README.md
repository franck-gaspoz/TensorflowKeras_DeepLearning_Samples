# Tensorflow Keras deep learning samples

Python library and PyCharm project that includes deep learning samples and neural networks visualization tools:

- single fully connected layer
- deep neural network with perceptron
- image classification with VGG16
- neural network model visualization
- Web Api that serves application features

## Screenshots

### a deep NN view

![Deep NN draw](doc/DeepNN.png)

### image prediction with WebAPI and VGG16

#### starting the server Uvicorn

![server logs](doc/server.png)

#### querying for a prediction giving a mug picture

![mug picture](doc/CNN-VGG-mug.jpg)

gives a very good result: probability to be a coffee mug > 98%

![mug prediction](doc/mug-prediction.png)

#### querying for a prediction giving a car picture "talbot samba"

![car picture](doc/talbot-samba-red.jpeg)

is very less accurate: probability to be a 'minibus' > 81% ;)

![car prediction](doc/samba-predict.png)

## Dependencies

- python 3.9.9
- tensorflow 2.7.0
- keras 2.7.0
- matplotlib 3.5.1
- shap 0.40.0
- pydot 1.4.2
- graphviz
- fastapi 0.72.0
- uvicorn[standard] 0.17
