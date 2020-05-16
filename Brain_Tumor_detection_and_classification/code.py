!unzip brain_tumor_dataset.zip -d data
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten , Dense
from fastai import *
from fastai.vision import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
classifier = Sequential()
classifier.add(Convolution2D(64,3,3 , input_shape = (256,256,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(64,3,3 ,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
from keras.layers import Flatten
classifier.add(Flatten())
classifier.add(Dense(output_dim = 256,activation='relu'))
classifier.add(Dense(output_dim = 128,activation='relu'))
classifier.add(Dense(output_dim = 64,activation='relu'))

classifier.add(Dense(output_dim=1,activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics = ['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
classifier.fit_generator(training_set,
                         samples_per_epoch = 213,
                         nb_epoch = 100,
                         validation_data = test_set_,
                         nb_val_samples = 40)
training_set = train_datagen.flow_from_directory(
        '/content/data/brain_tumor_dataset/training_set',
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary')

test_set_ = test_datagen.flow_from_directory(
        '/content/data/brain_tumor_dataset/test_set',
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary')

from keras.preprocessing import image
test_image = image.load_img('/content/220px-Normal_axial_T2-weighted_MR_image_of_the_brain.jpg', target_size = (256, 256))
test_image

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
if result[0][0] == 1: 
  print('yes Brain Tumor') 
else:
    print('No Brain Tumor')
