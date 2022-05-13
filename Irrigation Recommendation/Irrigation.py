#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
from sklearn import preprocessing,neighbors
import pandas as pd

from sklearn.model_selection import cross_val_score


# In[2]:



path ='Dataset\datasets.csv'
data=pd.read_csv(path)
data.replace('?',-99999,inplace=True)
data.columns=['CropType','cropDays','soilMoisture','temp','humidity','y']
print(data.head())


# In[3]:


x=np.array(data.drop(['y'],1))
y=np.array(data['y'])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
cls=neighbors.KNeighborsClassifier()
cls.fit(x_train,y_train)
accuracy=cls.score(x_test,y_test)
print(accuracy)


# In[26]:



prediction =cls.predict([[2,32,700,32,32]])
if(prediction==1):
    print("Irrigation is required")
else:
    print("Irrigation is not required")


# In[21]:



import pickle
pickle.dump(cls, open('model.pkl','wb'))


# In[27]:



model = pickle.load(open('model.pkl','rb'))
prediction=model.predict([[2,32,700,32,32]])
if(prediction==1):
    print("Irrigation is required")
else:
    print("Irrigation is not required")

