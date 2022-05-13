import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__,template_folder='template')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=["POST"])
def predict():

    int_features = [float(x) for x in request.form.values()]
    
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template('index.html', prediction_text= 'Fertlizer Needed is {}'.format(prediction))
    


if __name__ == "__main__":
    app.run(debug=True)