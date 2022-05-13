#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import preprocessing,neighbors


# In[2]:


data = pd.read_csv('data.csv')


# In[3]:


data.head(5)


# In[4]:


inputs = data[['pH','EC','OC','OM','N','P','K','Zn','Fe','Cu','Mn','Sand','Silt','Clay','CaCO3','CEC']]
fertile= data['Output']


# In[5]:


sns.heatmap(data.corr(),annot=True)


# In[6]:


from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(inputs,fertile,test_size = 0.2,random_state =0)


# In[7]:


from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier(n_estimators=20, random_state=0)
regressor.fit(Xtrain,Ytrain)

predicted_values = regressor.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)

print(x)

print(classification_report(Ytest,predicted_values))


# In[9]:


from sklearn.model_selection import cross_val_score
score = cross_val_score(regressor,inputs,fertile,cv=5)
score


# In[10]:


regressor.predict([[7.74,0.40,0.01,0.01,75,20.0,279,0.48,6.4,0.21,4.7,84.3,6.8,8.9,6.72,7.81]])


# In[14]:


import pickle
pickle.dump(regressor, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[7.74,0.40,0.01,0.01,75,20.0,279,0.48,6.4,0.21,4.7,84.3,6.8,8.9,6.72,7.81]]))


# In[ ]:




