from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
# Keras
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__,template_folder='template')





SoilNet = load_model('SoilNet-2.h5',compile=False)
print('Model loaded. Check http://127.0.0.1:5000/')
classes = {0:"Alluvial Soil:-{ Rice,Wheat,Sugarcane,Maize,Cotton,Soyabean,Jute }",1:"Black Soil:-{ Virginia, Wheat , Jowar,Millets,Linseed,Castor,Sunflower} ",2:"Clay Soil:-{ Rice,Lettuce,Chard,Broccoli,Cabbage,Snap Beans }",3:"Red Soil:{ Cotton,Wheat,Pilses,Millets,OilSeeds,Potatoes }"}


def model_predict(image_path, model):
    print("Predicted")
    image = load_img(image_path,target_size=(150,150))
    image = img_to_array(image)
    image = image/255
    image = np.expand_dims(image,axis=0)
    result = np.argmax(model.predict(image))
    prediction = classes[result]
    if result == 0:
        return("Alluvial Soil")
    elif result == 1:
        return("Black Soil")
    elif result == 2:
        return("Clay Soil")
    elif result == 3:
        return("Red Soil")
    


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])

def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result=(model_predict(file_path, SoilNet))
        return result
                   
        
    return None
if __name__ == '__main__':
    app.run(debug='true')
    


