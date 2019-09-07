from PIL import Image
from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def load_models():
    global models
    models = {
              'mnist'          : load_model('models/mnist_model.hdf5'),
              'dog_classifier' : load_model('models/dog-classifier.weights.best.inception3.hdf5')
              }

    global graph
    graph = tf.get_default_graph()

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
            data = request.files.get('image')
            img  = Image.open(data)
            img_arr = np.array(img, dtype='float32')
            return str(img_arr.shape)
        else:
            return "POST dog image."

if __name__ == "__main__":
    load_models()
    app.run(debug=True)
