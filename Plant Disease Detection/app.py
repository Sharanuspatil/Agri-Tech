from __future__ import division, print_function
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing import image
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import pickle
import cv2
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from os import listdir
from keras.preprocessing import image


import os, sys, glob, re
app = Flask(__name__,template_folder="templates")

model = pickle.load(open('cnn_model.pkl', 'rb'))
LabelBinarizers = pickle.load(open('label_transform.pkl', 'rb'))
default_image_size = tuple((256, 256))
model._make_predict_function()    
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

def model_predict(image_dir,model):
    im=convert_image_to_array(image_dir)
    np_image_li = np.array(im, dtype=np.float16) / 225.0
    npp_image = np.expand_dims(np_image_li, axis=0)
    result=model.predict(npp_image)
    itemindex = np.where(result==np.max(result))
    results = LabelBinarizers.classes_[itemindex[1][0]]
    return results

   
@app.route('/',methods=['GET'])
def index():
    return render_template('index2.html')
    

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        f = request.files['file']

  
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'static/user uploaded', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        pred = model_predict(file_path,model)
              
        return pred
    return None

if __name__ == '__main__':
    app.run(debug=True)

