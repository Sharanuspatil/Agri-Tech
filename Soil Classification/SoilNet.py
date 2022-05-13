#!/usr/bin/env python
# coding: utf-8

# In[51]:


#
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dropout, MaxPooling2D, AveragePooling2D, Dense, Flatten, Input, Conv2D, add, Activation
from tensorflow.keras.layers import (Dense, Dropout, Activation, Flatten, Reshape, Layer,
                          BatchNormalization, LocallyConnected2D,
                          ZeroPadding2D, Conv2D, MaxPooling2D, Conv2DTranspose,AveragePooling2D,
                          GaussianNoise, UpSampling2D, Input)

from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential , Model , load_model
from tensorflow.keras.preprocessing.image import load_img , img_to_array , ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from PIL import Image
import matplotlib.pyplot as plt

import cv2
from imutils import paths
import numpy as np
import os
import time
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


print("Tensorflow version: ",tf.__version__)


# In[4]:



train_dir = '/Users/Shreyu/Desktop/drone_agri/soil analysis/Soil_Dataset/Train'
test_dir = '/Users/Shreyu/Desktop/drone_agri/soil analysis/Soil_Dataset/Test'

image_size = 224


# In[ ]:


batch_size = 32

train_datagen = ImageDataGenerator(rescale = 1./255,
                            rotation_range=45,
                            zoom_range=0.40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.15,
                            horizontal_flip=True,
                            vertical_flip= True,
                            fill_mode="nearest")

train_data = train_datagen.flow_from_directory(train_dir,
                                              target_size=(150,150),
                                              batch_size=32,
                                              class_mode="categorical")


# In[ ]:


test_datagen = ImageDataGenerator(rescale = 1./255)

test_data = test_datagen.flow_from_directory(test_dir,
                                            target_size=(150,150),
                                            batch_size=32,
                                            class_mode="categorical")


# In[ ]:


train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                              validation_split = 0.2,
                                                              subset = "training",
                                                              seed = 42,
                                                              image_size = (150,150),
                                                              batch_size = 40)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                             validation_split = 0.2,
                                                             subset = "validation",
                                                             seed = 42,
                                                             image_size = (150,150),
                                                             batch_size = 40)


# In[ ]:


## Configuring dataset for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE
training_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
testing_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


# In[ ]:



model = Sequential(name="SoilNet")
model.add(Conv2D(64,(3,3),activation = "relu",padding ="same",kernel_initializer="he_normal", input_shape=(150,150,3)))
#model.add(tf.keras.layers.LeakyReLU())
#model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),activation = "relu",padding ="same",kernel_initializer="he_normal"))
#model.add(tf.keras.layers.LeakyReLU())
model.add(BatchNormalization())

model.add(AveragePooling2D(pool_size = (2,2), strides=2))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Conv2D(128,(3,3),activation = "relu",padding ="same",kernel_initializer="he_normal"))
#model.add(tf.keras.layers.LeakyReLU())
#model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),activation = "relu",padding ="same",kernel_initializer="he_normal"))
#model.add(tf.keras.layers.LeakyReLU())
model.add(BatchNormalization())

model.add(AveragePooling2D(pool_size = (2,2), strides=2))
model.add(Dropout(0.5))
model.add(BatchNormalization())

#lk = tf.keras.layers.LeakyReLU()
model.add(Conv2D(256,(3,3),activation = "relu", padding ="same",kernel_initializer="he_normal"))
#model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),activation = "relu",padding ="same",kernel_initializer="he_normal"))
#model.add(tf.keras.layers.LeakyReLU())
model.add(BatchNormalization())

model.add(AveragePooling2D(pool_size = (2,2), strides=2))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(512,activation="relu"))
model.add(Dropout(0.7))
model.add(Dense(4,activation="softmax"))

opt = RMSprop(learning_rate = 0.0001, rho = 0.99, epsilon = 1e-08, decay = 0.0)
model.compile(optimizer=opt,loss="categorical_crossentropy",metrics=["accuracy"])

reduction_lr = ReduceLROnPlateau(monitor = "val_accuracy",patience = 2 ,verbose = 1, factor = 0.3, min_lr = 0.0000001)
reduction_lr1 = ReduceLROnPlateau(monitor = "val_loss",patience = 2 ,verbose = 1, factor = 0.3, min_lr = 0.0000001)


# In[ ]:


#bot_callback = botCallback(access_token)
#plotter = Plotter(access_token)
#callback_list = [bot_callback,plotter] callbacks=callback_list

start = time.time()

history = model.fit_generator(train_data,
                    validation_data = test_data,
                    epochs=20,
                    callbacks = [reduction_lr,reduction_lr1])
end = time.time()
print("Total train time: ",(end-start)/60," mins")


# In[ ]:


"""
model = tf.keras.models.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(128, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(2, activation= 'softmax')
])
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
"""


# In[ ]:


"""
#=================================================================
chanDim = 1
model = Sequential(name="SoilNet")
model.add(Conv2D(32, (3, 3), padding="same",input_shape=(150,150,3)))
model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (4, 4), padding="same"))
model.add(Activation("relu"))

model.add(Conv2D(64, (4, 4), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (4, 4), padding="same"))
model.add(Activation("relu"))

model.add(Conv2D(128, (4, 4), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation("softmax"))


model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
reduction_lr = ReduceLROnPlateau(monitor = "val_accuracy",patience = 2 ,verbose = 1, factor = 0.2, min_lr = 0.00001)
callback_list = [reduction_lr]
model.summary()
plot_model(model,show_shapes=True)
"""


# In[ ]:


"""
history = model.fit(train_ds,
                    validation_data = test_ds,
                    epochs=5)
"""


# In[ ]:


def plot_graph(history,string):
    plt.figure(figsize=(12,8))
    plt.plot(history.history[string],label=str(string))
    plt.plot(history.history["val_"+str(string)],label="val_"+str(string))
    plt.xlabel("Epochs")
    plt.ylabel(str(string))
    plt.legend()
    plt.show()
plot_graph(history,"accuracy")
plot_graph(history,"loss")


# In[ ]:


model.save("SoilNet.h5")


# In[ ]:


from IPython.display import FileLink
FileLink('SoilNet.h5')


# In[69]:


model_path=r'/Users/Shreyu/Downloads/SoilNet.h5'
model = tf.keras.models.load_model(model_path)

# In[70]:


image_path='/Users/Shreyu/Desktop/drone_agri/soil analysis/Soil_Dataset/Test/Black_Soil/Black_4.jpg'


# In[71]:


SoilNet = load_model(model_path)

classes = {0:"Alluvial Soil:-{ Rice,Wheat,Sugarcane,Maize,Cotton,Soyabean,Jute }",1:"Black Soil:-{ Virginia, Wheat , Jowar,Millets,Linseed,Castor,Sunflower} ",2:"Clay Soil:-{ Rice,Lettuce,Chard,Broccoli,Cabbage,Snap Beans }",3:"Red Soil:{ Cotton,Wheat,Pilses,Millets,OilSeeds,Potatoes }"}

def model_predict(image_path,model):
    print("Predicted")
    image = load_img(image_path,target_size=(150,150))
    image = img_to_array(image)
    image = image/255
    image = np.expand_dims(image,axis=0)
    
    result = np.argmax(model.predict(image))
    prediction = classes[result]
    
    print(prediction)
    if result == 0:
        print("Alluvial")
        
    elif result == 1:
        print("Black")
    
    elif result == 2:
        print("Clay")
        
    elif result == 3:
        print("Red")
        

    


# In[67]:


model_predict(image_path,model)


# In[ ]:
# save the model to disk
import os
path = "w:/tmp/autoencoder5-13/"
os.mkdir(path)


