# this part will prevent tensorflow to allocate all the avaliable GPU Memory

import tensorflow as tf
from keras import backend as k
import shutil
import os
import tarfile
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.manifold import TSNE
from sklearn import preprocessing
import pandas as pd
from multiprocessing import Process# this is used for multithreading
import multiprocessing
import codecs# this is used for file operations 
import random as r
import time
import math
import cv2
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import log_loss,confusion_matrix
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D,MaxPooling2D,Conv2D,Dropout,BatchNormalization
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.utils import np_utils
from keras.preprocessing import image
from sklearn.datasets import load_files
import cv2
import pickle
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,precision_score,recall_score
from statistics import mode
from sklearn.utils.multiclass import unique_labels
# Don't pre-allocate memory; allocate as-needed
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))

from keras.callbacks import ReduceLROnPlateau
from keras import regularizers
# Training configuaration for CNN- to be trained on whole images
img_width, img_height = 224,224
train_dir = "train/Whole"
val_dir = "val/Whole"
test_dir="test"
nb_train_samples = 320000
nb_validation_samples = 40000 
batch_size = 50
epochs = 20
num_classes=16
log=[]

# preparing Training and Validation data using ImageDataGenerator
train_datagen = ImageDataGenerator(
rescale = 1./255,
featurewise_std_normalization=True)

val_datagen = ImageDataGenerator(
rescale = 1./255,
featurewise_std_normalization=True)

train_generator = train_datagen.flow_from_directory(
train_dir,
target_size = (img_height, img_width),
batch_size = batch_size, 
class_mode = "categorical")

val_generator = val_datagen.flow_from_directory(
val_dir,
target_size = (img_height, img_width),
batch_size = batch_size, 
class_mode = "categorical")

# Importing keras InceptionRsnetv2 pretrained model (on ImageNet)
# We will use the ImageNet weight to initialize the model and fine tune using backpropagation(Transfer Learning) 

model=applications.InceptionResNetV2(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
#for layer in model.layers[:15]:
#    layer.trainable = False
#Adding custom Layers 
x = model.output
x=Dropout(0.5)(x)
x = Flatten()(x)
output = Dense(num_classes, activation="softmax")(x)
model = Model(input = model.input, output = output)


#model.compile(loss = "categorical_crossentropy", optimizer = 'adam', metrics=["accuracy"])
checkpoint = ModelCheckpoint("inceptionresnet.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=3, min_lr=0.0001,mode='auto')
callbacks=[checkpoint,reduce_lr]
history=model.fit_generator(
train_generator,
samples_per_epoch =nb_train_samples//batch_size,
epochs = epochs,
validation_data = val_generator,
validation_steps =math.ceil(nb_validation_samples//(batch_size)),
callbacks = callbacks,verbose=1)
log.append(history)
model.save('Inceptionresnet_final.h5')

model=applications.VGG16(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
#for layer in model.layers[:15]:
#    layer.trainable = False
#Adding custom Layers 
x = model.output
x = Flatten()(x)
output = Dense(num_classes, activation="softmax")(x)
model = Model(input = model.input, output = output)
checkpoint = ModelCheckpoint("vgg16.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=3, min_lr=0.0001,mode='auto')
callbacks=[checkpoint,reduce_lr]
history=model.fit_generator(
train_generator,
samples_per_epoch =nb_train_samples//batch_size,
epochs = epochs,
validation_data = val_generator,
validation_steps =math.ceil(nb_validation_samples//(batch_size)),
callbacks = callbacks,verbose=1)
model.save('vgg16_final.h5')