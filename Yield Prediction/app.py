import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('model1.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    com_fea=['Crop_Jack Fruit', 'Crop_Black pepper', 'State_Name_Karnataka',
       'Season_Rabi', 'Crop_Wheat', 'State_Name_Chhattisgarh',
       'Crop_Groundnut', 'Season_Autumn     ', 'State_Name_Odisha',
       'State_Name_Tamil Nadu', 'State_Name_West Bengal', 'Crop_Potato',
       'Crop_Cotton(lint)', 'Crop_Other Kharif pulses', 'Crop_Safflower',
       'Area', 'State_Name_Nagaland', 'Crop_Arhar/Tur',
       'State_Name_Uttarakhand', 'Crop_Linseed', 'Crop_Maize',
       'State_Name_Chandigarh', 'State_Name_Mizoram', 'Crop_Onion',
       'Crop_Cardamom', 'Crop_Dry chillies', 'Crop_Horse-gram',
       'State_Name_Andhra Pradesh', 'State_Name_Manipur', 'Crop_Bajra',
       'State_Name_Uttar Pradesh', 'Crop_Soyabean', 'Season_Winter     ',
       'Crop_other oilseeds', 'Crop_Peas & beans (Pulses)',
       'State_Name_Haryana', 'Crop_Rice', 'Crop_Niger seed', 'Crop_Banana',
       'Crop_Sesamum', 'Crop_Jute', 'Crop_Cabbage', 'Crop_Moong(Green Gram)',
       'State_Name_Puducherry', 'State_Name_Himachal Pradesh', 'Crop_Mesta',
       'State_Name_Gujarat', 'State_Name_Madhya Pradesh',
       'Crop_Rapeseed &Mustard', 'Crop_Garlic', 'State_Name_Telangana ',
       'Crop_Dry ginger', 'Crop_Blackgram', 'Crop_Cashewnut',
       'Season_Whole Year ', 'State_Name_Andaman and Nicobar Islands',
       'Season_Summer     ', 'State_Name_Goa', 'State_Name_Arunachal Pradesh',
       'Crop_Coconut ', 'Crop_Masoor', 'Crop_Castor seed',
       'State_Name_Rajasthan', 'Crop_Urad', 'State_Name_Maharashtra',
       'State_Name_Jammu and Kashmir ', 'Crop_Pump Kin', 'Crop_Sunflower',
       'Crop_Ragi', 'Crop_Coriander', 'State_Name_Bihar', 'Crop_Guar seed',
       'Crop_Other  Rabi pulses', 'Crop_Small millets', 'Crop_Khesari',
       'Crop_Arecanut', 'Crop_Other Cereals & Millets', 'State_Name_Kerala',
       'Crop_Cowpea(Lobia)', 'Crop_Jowar', 'Season_Kharif     ',
       'Crop_Sugarcane', 'Crop_Turmeric', 'Crop_Gram', 'State_Name_Punjab',
       'Crop_Barley', 'Crop_Tapioca', 'State_Name_Assam', 'Crop_Tobacco',
       'Crop_Sannhamp', 'Crop_Moth', 'Crop_Sweet potato']


    dataf=pd.DataFrame(columns=com_fea)
    dataf.loc[len(dataf)]=0
    
    
    features = [x for x in request.form.values()] #state | season | crop | area
    dataf['Area']=float(features[3])
    
    for j in com_fea:
        test_list=j.strip().split('_')
        
        if (test_list[0]=='State'):
            if(test_list[-1]==features[0]):
                dataf[j]=1
                
        elif (test_list[0]=='Season'):
            if(test_list[-1]==features[1]):
                dataf[j]=1
                
        elif (test_list[0]=='Crop'):            
            if(test_list[-1]==features[2]):
                dataf[j]=1
                    
    
    prediction = model.predict(dataf)
    print(prediction)
    output = round(prediction[0], 2)


    return render_template('index.html', prediction_text='Predicted rate = {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug='true')