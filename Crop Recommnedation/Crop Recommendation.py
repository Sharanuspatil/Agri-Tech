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


data = pd.read_csv('Crop.csv')


# In[3]:


data.head()


# In[4]:


data.size


# In[5]:


data ['label'].unique()


# In[6]:



inputs = data[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
crop = data['label']


# In[7]:


from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(inputs,crop,test_size = 0.2,random_state =2)


# In[8]:


from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier(n_estimators=20, random_state=0)
regressor.fit(Xtrain,Ytrain)

predicted_values = regressor.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)

print(x)

print(classification_report(Ytest,predicted_values))


# In[9]:


from sklearn.model_selection import cross_val_score
score = cross_val_score(regressor,inputs,crop,cv=5)
print(score)


# In[10]:


regressor.predict([[73,57,44,20.87974371,82.00274423,6.502985292,202.9355362]])


# In[11]:


import pickle
pickle.dump(regressor, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[73,57,44,20.87974371,82.00274423,6.502985292,202.9355362]]))


# In[ ]:




