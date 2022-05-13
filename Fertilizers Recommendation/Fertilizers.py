#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree


# In[2]:


data = pd.read_csv('Fertilizer Prediction.csv')


# In[3]:


data.head()


# In[4]:


data.size


# In[5]:


data ['Fertilizer Name'].unique()


# In[6]:


data ['Soil Type'].unique()


# In[7]:


from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
data["Soil Types"] = lb_make.fit_transform(data["Soil Type"])
data[["Soil Type", "Soil Types"]].head


# In[8]:


data ['Crop Type'].unique()


# In[9]:


from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
data["Crop Types"] = lb_make.fit_transform(data["Crop Type"])
data[["Crop Type", "Crop Types"]].head


# In[10]:



inputs = data[['Temparature','Humidity ','Moisture','Soil Types','Crop Types','Nitrogen','Potassium','Phosphorous']]
crop = data['Fertilizer Name']
print(data['Crop Types'])
print(data['Soil Types'])


# In[11]:


sns.heatmap(data.corr(),annot=True)


# In[12]:


from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(inputs,crop,test_size = 0.2,random_state =0)


# In[13]:


from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier(n_estimators=20, random_state=0)
regressor.fit(Xtrain,Ytrain)

predicted_values = regressor.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)

print(x)

print(classification_report(Ytest,predicted_values))


# In[14]:


from sklearn.model_selection import cross_val_score
score = cross_val_score(regressor,inputs,crop,cv=5)
print(score)


# In[15]:


regressor.predict([[26,52,38,4,3,37,0,0]])
regressor.predict([[30,60,27,3,9,4,17,17]])


# In[16]:


import pickle
pickle.dump(regressor, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[26,52,38,4,3,37,0,0]]))

