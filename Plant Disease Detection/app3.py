from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
import pickle
import cv2
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt

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
    return render_template('index.html')



@app.route('/predict',methods=['GET','POST'])
def predict():
    print("Entered")
    if request.method == 'POST':
        print("Entered here")
        file = request.files['image'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred = model_predict(file_path,model)
        print(pred)     
        return render_template( 'index.html',pred_output = pred)
    


if __name__ == '__main__':
    app.run(debug=True,threaded=False)