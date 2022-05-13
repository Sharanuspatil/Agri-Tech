#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk(r'H:\New folder (7)'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        break


# In[2]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf 
import keras 
from keras.layers import Conv2D
from keras.models import Sequential
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import cv2
import re
import random
random.seed(0)
np.random.seed(0)


# In[3]:


wheat = plt.imread("C:\\Users\\DELL\\Desktop\\Test\\kag2\\wheat\\wheat0004a.jpeg")
jute = plt.imread("C:\\Users\\DELL\\Desktop\\Test\\kag2\\jute\\jute005a.jpeg")
cane = plt.imread("C:\\Users\\DELL\\Desktop\\Test\\kag2\\sugarcane\\sugarcane0010arot.jpeg")
rice = plt.imread("C:\\Users\\DELL\\Desktop\\Test\\kag2\\rice\\rice032ahs.jpeg")
maize = plt.imread("C:\\Users\\DELL\\Desktop\\Test\\kag2\\maize\\maize003a.jpeg")
plt.figure(figsize=(20,3))
plt.subplot(1,5,1)
plt.imshow(jute)
plt.title("jute")
plt.subplot(1,5,2)
plt.imshow(maize)
plt.title("maize")
plt.subplot(1,5,3)
plt.imshow(rice)
plt.title("rice")
plt.subplot(1,5,4)
plt.imshow(cane)
plt.title("sugarcane")
plt.subplot(1,5,5)
plt.imshow(wheat)
plt.title("wheat")


# In[4]:


jutepath = "C:\\Users\\DELL\\Desktop\\Test\\kag2\\jute"
maizepath = "C:\\Users\\DELL\\Desktop\\Test\\kag2\\maize"
ricepath = "C:\\Users\\DELL\\Desktop\\Test\\kag2\\rice"
sugarcanepath= "C:\\Users\\DELL\\Desktop\\Test\\kag2\\sugarcane"
wheatpath = "C:\\Users\\DELL\\Desktop\\Test\\kag2\\wheat"

jutefilename = os.listdir(jutepath)
maizefilename = os.listdir(maizepath)
ricefilename = os.listdir(ricepath)
sugarcanefilename = os.listdir(sugarcanepath)
wheatfilename = os.listdir(wheatpath)

X= []


# In[5]:


for fname in jutefilename:
    X.append([os.path.join(jutepath,fname),0])
for fname in maizefilename:
    X.append([os.path.join(maizepath,fname),1])
for fname in ricefilename:
    X.append([os.path.join(ricepath,fname),2])
for fname in sugarcanefilename:
    X.append([os.path.join(sugarcanepath,fname),3]) 
for fname in wheatfilename:
    X.append([os.path.join(wheatpath,fname),4])  
X = pd.DataFrame(X,columns = ['path','labels'])    


# In[6]:


X.head()


# In[7]:



ohencoder = OneHotEncoder(handle_unknown='ignore',sparse=False)
ohlabel = pd.DataFrame(ohencoder.fit_transform(X[['labels']]),dtype = 'int64',columns = ['label0','label1','label2','label3','label4'])
label_X = X.copy()
X = pd.concat([X,ohlabel],axis = 1)
new_X = X.drop(['labels'],axis = 1)


# In[8]:


train,test = train_test_split(label_X,test_size=0.2,random_state=0,shuffle = True)


# In[9]:


X_train = train['path'].values
y_train = train.drop(['path'],axis=1).values
X_test = test['path'].values
y_test = test.drop(['path'],axis=1).values


# In[10]:


def flat_x(data):
    flat = []
    for i in data:
        img = plt.imread(i)
        img = img/255.
        flat.append(img.reshape([1,-1]))
    flat =  np.array(flat)    
    flat = flat.reshape(-1,224*224*3)       
    return flat
def flat_x_oned(data):
    data = flat_x(data)
    data  = data[:,:224*224]
    return flat


# In[11]:


flat_X_train = flat_x(X_train)
flat_X_test = flat_x(X_test)


# In[12]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_pca=pca.fit_transform(flat_X_train)
print(flat_X_train.shape)
print(X_train_pca.shape)


# In[13]:


import seaborn as sns
plt.figure(figsize=(15,8))
sns.scatterplot(X_train_pca[:,0],X_train_pca[:,1],hue = np.ravel(y_train),palette='Paired_r',s=80)
plt.title("PCA decomposition of 150528 columns to 2 columns")
plt.grid()


# In[14]:


from sklearn.manifold import TSNE
tsne=TSNE(n_components=2,random_state=0)
X_train_tsne=tsne.fit_transform(flat_X_train[:,:224*224])
print(flat_X_train.shape)
print(X_train_pca.shape)


# In[15]:


plt.figure(figsize=(15,8))
sns.scatterplot(X_train_tsne[:,0],X_train_tsne[:,1],hue = np.ravel(y_train),palette='Paired_r',s=80)
plt.title("2d scatter plot of one dimensions of images using tsne")
plt.grid()


# In[16]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(flat_X_train,np.ravel(y_train))


# In[17]:


lr.score(flat_X_test,np.ravel(y_test))


# In[18]:


def modelpipeline(imagepath,model = lr,label=-1):
    pdict = {0:"jute",1:"maize",2:"rice",3:"sugarcane",4:"wheat"}
    pred_x = flat_x([imagepath])
    pred = model.predict(pred_x)
    plt.imshow(plt.imread(imagepath))
    if (label!=-1):
        plt.title("prediction : {} \naccurate  : {}".format(pdict[pred[0]],pdict[label]))
    else:
        plt.title("prediction : {}".format(pdict[pred[0]]))


# In[19]:


modelpipeline('C:\\Users\\DELL\\Desktop\\Test\\kag2\\rice\\rice024ahs.jpeg')


# In[20]:


modelpipeline('C:\\Users\\DELL\\Desktop\\Test\\kag2\\wheat\\wheat0004a.jpeg')


# In[21]:


modelpipeline("C:\\Users\\DELL\\Desktop\\Test\\kag2\\maize\\maize008ahf.jpeg")


# In[22]:


modelpipeline("C:\\Users\\DELL\\Desktop\\Test\\kag2\\jute\\jute005a.jpeg")


# In[23]:


plt.figure(figsize=(20,20))
for num,path in enumerate(X_test[0:20]):
    plt.subplot(4,5,num+1)
    modelpipeline(path,lr,y_test[num][0])


# In[24]:


plt.figure(figsize=(20,20))
for num,path in enumerate(X_test[20:40]):
    plt.subplot(4,5,num+1)
    modelpipeline(path,lr,y_test[num+20][0])


# In[25]:


plt.figure(figsize=(20,20))
for num,path in enumerate(X_test[40:60]):
    plt.subplot(4,5,num+1)
    modelpipeline(path,lr,y_test[num+40][0])


# In[26]:


import joblib
filename = 'lr_model.sav'
joblib.dump(lr, filename)


# In[ ]:




