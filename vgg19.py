from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt


# re-size all the images to this
IMAGE_SIZE = [224, 224]


train_path="E:/Drishti Vgg16/Dataset1/train"
test_path="E:/Drishti Vgg16/Dataset1/test"
val_path="E:/Drishti Vgg16/Dataset1/valid"


x_train=[]

for folder in os.listdir(train_path):
    sub_path=train_path+"/"+folder
    for img in os.listdir(sub_path):
        image_path=sub_path+"/"+img
        img_arr=cv2.imread(image_path)
        img_arr=cv2.resize(img_arr,(224,224))
        x_train.append(img_arr)
        
x_test=[]

for folder in os.listdir(test_path):
    sub_path=test_path+"/"+folder
    for img in os.listdir(sub_path):
        image_path=sub_path+"/"+img
        img_arr=cv2.imread(image_path)
        img_arr=cv2.resize(img_arr,(224,224))
        x_test.append(img_arr)
        
x_val=[]

for folder in os.listdir(val_path):
    sub_path=val_path+"/"+folder
    for img in os.listdir(sub_path):
        image_path=sub_path+"/"+img
        img_arr=cv2.imread(image_path)
        img_arr=cv2.resize(img_arr,(224,224))
        x_val.append(img_arr)
        
        
train_x=np.array(x_train)
test_x=np.array(x_test)
val_x=np.array(x_val)


train_x.shape,test_x.shape,val_x.shape

train_x=train_x/255.0
test_x=test_x/255.0
val_x=val_x/255.0


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
val_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 2,
                                                 class_mode = 'sparse')

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (224, 224),
                                            batch_size = 2,
                                            class_mode = 'sparse')

val_set = val_datagen.flow_from_directory(val_path,
                                            target_size = (224, 224),
                                            batch_size = 2,
                                            class_mode = 'sparse')


training_set.class_indices

train_y=training_set.classes
test_y=test_set.classes
val_y=val_set.classes

train_y.shape,test_y.shape,val_y.shape


vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


for layer in vgg.layers:
    layer.trainable = False
    

x = Flatten()(vgg.output)

prediction = Dense(3, activation='softmax')(x)

#create a model object
model = Model(inputs=vgg.input, outputs=prediction)

#view the structure of the model
model.summary()

prediction = Dense(3, activation='softmax')(x)


model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer="adam",
  metrics=['accuracy']
)


from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)

#Early stopping to avoid overfitting of model


#fitting the model
history = model.fit(
  train_x,
  train_y,
  validation_data=(val_x,val_y),
  epochs=30,
  callbacks=[early_stop],
  batch_size=32,shuffle=True)
