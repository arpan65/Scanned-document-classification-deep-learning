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
from keras.models import load_model
from statistics import mode
from sklearn.utils.multiclass import unique_labels

model1=load_model('vgg16.h5')
model2=load_model('resinception.h5')
img_width, img_height = 224,224
train_dir = "train"
val_dir = "val"
test_dir="test"
nb_train_samples = 300000
nb_validation_samples = 32000
batch_size = 1280
epochs = 20
num_classes=16
log=[]
# Test Data generator to generate test data in same format of training data
test_datagen = ImageDataGenerator(
rescale = 1./255,
featurewise_std_normalization=True)
test_generator = test_datagen.flow_from_directory(
test_dir,
target_size = (img_height, img_width),
batch_size = batch_size,shuffle='True',
class_mode = "categorical")
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,precision_score,recall_score
from statistics import mode
from sklearn.utils.multiclass import unique_labels
def eval_model(model1,model2,test_generator,limit):
    """
    This function creates an ensamble of candidate CNN models and predict the result
    model1,model2,model3 -- candidate models
    test_generator -- Instance of ItemDataGenerator to randomly generate test data
    limit -- limits the number of test samples
    
    """
    try:
         x,y=test_generator.next()
    except:
        print('')
    y_pred=[]
    y_true=[]
    count=0
    #print(len(x))
    for i in range(len(x)):
        #pred1=np.argmax(model1.predict(np.expand_dims(x[i], axis=0)))
        #pred2=np.argmax(model2.predict(np.expand_dims(x[i], axis=0)))
        #pred3=np.argmax(model3.predict(np.expand_dims(x[i], axis=0)))
        try:
            # If there are clear majority in prediction add the majority voted class
            y_pred.append(np.argmax(model1.predict(np.expand_dims(x[i], axis=0))+model2.predict(np.expand_dims(x[i], axis=0))))
        except:
            # else select prediction of second model 
            y_pred.append(pred2)
        y_true.append(np.argmax(y[i]))
        count+=1
        # limit the number of samples to be considered
        if(count==limit):
            break
    return y_pred,y_true

def get_result(y_pred,y_true):
    """
    This function evaluate the performance(accuracy,f1-score,precision,recall,confusion matrix of the network).
    """
    print('-----Model Evaluation------')
    accuracy=np.round(accuracy_score(y_pred,y_true),3)
    f1=np.round(f1_score(y_pred,y_true,average='macro'),3)
    precision=np.round(precision_score(y_pred,y_true,average='macro'),3)
    recall=np.round(recall_score(y_pred,y_true,average='macro'),3)
    print('Accuracy:',accuracy*100,'%')
    print('Macro F1 Score:',f1)
    print('Precision Score:',precision)
    print('Recall Score:',recall)
    #plot_confusion_matrix(y_true,y_pred)
    return accuracy,f1,precision,recall
    
y_pred,y_true=eval_model(model1,model1,test_generator,30000)
accuracy,f1,precision,recall=get_result(y_pred,y_true)	