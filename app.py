from PIL import Image
from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.resnet50 import ResNet50, decode_predictions
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import tensorflow as tf
import numpy as np
import pickle
import cv2
import os

app = Flask(__name__)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def load_models():
    global models
    models = {
              'mnist'            : load_model('models/mnist_model.hdf5'),
              'breed_classifier' : load_model('models/dog-classifier.weights.best.inception3.hdf5'),
              'dog_classifier'   : ResNet50(weights='imagenet'),
              'face_classifier'  : cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml'),
              }

    global graph
    graph = tf.get_default_graph()

def is_human(img):
    model = models['face_classifier']
    faces = model.detectMultiScale(img)

    return len(faces) > 0

def is_dog(img):
    model = models['dog_classifier']

    img = cv2.resize(img, (224, 224))
    img_4d = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
    pred_label = np.argmax(model.predict(img_4d))

    return ((pred_label >= 151) & (pred_label <= 268))

def get_dog_breed(img):
    img = cv2.resize(img, (224, 224))
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    img = InceptionV3(weights='imagenet', include_top=False).predict(preprocess_input(img))
    model = models['breed_classifier']
    pred  = model.predict(img)
    index = np.argmax(pred)

    with open('dog_names.pkl', 'rb') as pf:
        dog_names = pickle.load(pf)

    return dog_names[index]

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

            # Grayscale image
            img_gray  = load_img(data, color_mode='grayscale')
            # Color image
            img_color = load_img(data)

            arr_gray  = img_to_array(img_gray, dtype='uint8')
            arr_color = img_to_array(img_color)

            human = is_human(arr_gray)
            dog   = is_dog(arr_color)
            breed = get_dog_breed(arr_color)

            return str(breed)

        else:
            return "POST dog image."

if __name__ == "__main__":
    load_models()
    app.run(debug=True)
