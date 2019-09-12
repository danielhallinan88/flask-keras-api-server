from PIL import Image
from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.resnet50 import ResNet50
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def load_models():
    global models
    models = {
              'mnist'            : load_model('models/mnist_model.hdf5'),
              'dog_classifier'   : load_model('models/dog-classifier.weights.best.inception3.hdf5'),
              'breed_classifier' : '',
              'face_classifier'  : cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml'),
              }

    global graph
    graph = tf.get_default_graph()

def is_human(gray):
    model = models['face_classifier']
    faces = model.detectMultiScale(gray)

    return len(faces) > 0

def is_dog(img_path):
    pass

@app.route('/')
@app.route('/test', methods=["GET","POST"])
def test():
    if request.method == "POST":
        return test_string
    else:
        if 'num' in request.args.keys():
            return request.args['num']
        else:
            return "Test Succeeded!"

@app.route('/mnist', methods=["GET","POST"])
def mnist():
    model = models['mnist']

    with graph.as_default():
        if request.method == "POST":
            data = request.files.get("image")
            img        = Image.open(data)
            img_arr    = np.array(img, dtype='float32')
            img_arr_sh = img_arr.reshape(1, 28, 28, 1)
            prediction = model.predict(img_arr_sh)
            result     = np.argmax(prediction)
            return str(result)
        else:
            return "Please POST an image."

@app.route('/dog-classifier', methods=['GET', 'POST'])
def dog_classify():

    with graph.as_default():
        model = models['dog_classifier']

        if request.method == 'POST':
            # Preprocess image
            data = request.files.get('image')
            # Convert image to grayscale
            #img     = Image.open(data)
            #gray    = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
            #gray    = np.array(gray, dtype='uint8')
            img     = load_img(data, color_mode='grayscale')
            img_arr = img_to_array(img, dtype='uint8')
            human   = is_human(img_arr)
            return str(human)

        else:
            return "POST dog image."

if __name__ == "__main__":
    load_models()
    app.run(debug=True)
