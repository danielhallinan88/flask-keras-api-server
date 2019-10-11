from keras.models import load_model
from keras.datasets import mnist
from PIL import Image
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


model = load_model('models/mnist_model.hdf5')
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def make_samples():
    samples = x_test[0:10]
    for i, sample in enumerate(samples):
        im = Image.fromarray(sample)
        im.save('images/test_image_{}.jpg'.format(i))

def predict(sample):
    sample = sample.reshape(1, 28, 28, 1)
    prediction = model.predict(sample)
    print(np.argmax(prediction))

def predict_all():
    for i in range(10):
        image_path = 'images/test_image_{}.jpg'.format(i)
        image      = Image.open(image_path)
        img_arr    = np.array(image)
        predict(img_arr)

image_path = 'images/test_image_0.jpg'
image      = Image.open(image_path)
img_arr    = np.array(image, dtype='float32')
print(img_arr.dtype)
