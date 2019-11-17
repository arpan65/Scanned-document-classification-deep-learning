import warnings
warnings.filterwarnings("ignore")
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


classes = {'0': 'letter',
 '1': 'form',
 '10': 'budget',
 '11': 'invoice',
 '12': 'presentation',
 '13': 'questionnaire',
 '14': 'resume',
 '15': 'memo',
 '2': 'email',
 '3': 'handwritten',
 '4': 'advertisement',
 '5': 'scientific report',
 '6': 'scientific publication',
 '7': 'specification',
 '8': 'file folder',
 '9': 'news article'}
 label_path='labels/'
# This function will create an index file with image paths and labels in csv format
# obtained csv will have two columns path(path to acess the image),label(class of the image)
def create_csv(file_path):
    images=[]
    labels=[]
    label_file=pd.DataFrame(columns=['path','label'],index=None)
    file=open(file_path)
    for ele in file:
        images.append(ele.split(' ')[0])
        labels.append(ele.split(' ')[1].rstrip())
    label_file.path=images
    label_file.label=labels
    return label_file

train_label_path=label_path+'train.txt' 
test_label_path=label_path+'test.txt' 
val_label_path=label_path+'val.txt'
# creating index file for train,test and val dataset
train_label=create_csv(train_label_path)
test_label=create_csv(test_label_path)
val_label=create_csv(val_label_path)
    
# Creating train,validation and Test directories
os.mkdir('train')
os.mkdir('test')
os.mkdir('val')
# run this cell to enable intra domain transfer learning
regions = ['Whole','header', 'footer', 'left_body', 'right_body']
for region in regions:
    os.mkdir('train/'+region)
    os.mkdir('val/'+region)
classes_name=[v for (k,v) in classes.items()]
for region in regions:
    for label in classes_name:
        os.mkdir('train/'+region+'/'+label)
        os.mkdir('val/'+region+'/'+label)
# Moving files for test directory from common directory
folders=[classes[str(i)] for i in range(16)]
for folder in folders:
    if(os.path.exists('test/'+folder)== False):
        os.mkdir('test/'+folder)
for i in tqdm(range(len(test_label))):
    if(os.path.exists('images/'+test_label.iloc[i].path)):
        source='images/'+test_label.iloc[i].path
        dest='test/'+classes[str(test_label.iloc[i].label)]
        if(os.path.exists(dest+'/'+source.split('/')[-1])==False):
            shutil.move(source,dest)
 
# # Moving files for train directory from common directory
for i in tqdm(range(len(train_label))):
    if(os.path.exists('images/'+train_label.iloc[i].path)):
        source='images/'+train_label.iloc[i].path
        dest='train/Whole/'+classes[str(train_label.iloc[i].label)]
        if(os.path.exists(dest+'/'+source.split('/')[-1])==False):
            shutil.move(source,dest)

#  Moving files for validation directory from common directory
for i in tqdm(range(len(val_label))):
    if(os.path.exists('images/'+val_label.iloc[i].path)):
        source='images/'+val_label.iloc[i].path
        dest='val/Whole/'+classes[str(val_label.iloc[i].label)]
        if(os.path.exists(dest+'/'+source.split('/')[-1])==False):
            shutil.move(source,dest)
val_dir='val/Whole'
count,limit=0,2000
for folder in tqdm(os.listdir(val_dir)):
  #os.shuffle(val_dir+'/'+folder)
  for image in os.listdir(val_dir+'/'+folder):
        img = cv2.imread(val_dir+'/'+folder+'/'+image,0)
        height = img.shape[0]
        width = img.shape[1]
        header = img[0:(int(height*0.33)), 0:width]
        footer = img[int(height*0.67):height, 0:width]
        left_body = img[int(height*0.33):int(height*0.67), 0:int(width*0.5)]
        right_body = img[int(height*0.33):int(height*0.67), int(width*0.5):width]
        header_path='val/header/'+folder+'/'+image.split('/')[-1]
        footer_path='val/footer/'+folder+'/'+image.split('/')[-1]
        left_path='val/left_body/'+folder+'/'+image.split('/')[-1]
        right_path='val/right_body/'+folder+'/'+image.split('/')[-1]
        if(os.path.exists(header_path)==False):
            cv2.imwrite(header_path, header)
        if(os.path.exists(footer_path)==False):
            cv2.imwrite(footer_path, footer)
        if(os.path.exists(left_path)==False):
            cv2.imwrite(left_path, left_body)
        if(os.path.exists(right_path)==False):
            cv2.imwrite(right_path, right_body)
        count+=1
        if(count==limit):
            break

train_dir='train/Whole'
count,limit=0,2000
for folder in tqdm(os.listdir(train_dir)):
  #os.shuffle(train_dir+'/'+folder)
  count=0
    for image in os.listdir(train_dir+'/'+folder):
        img = cv2.imread(train_dir+'/'+folder+'/'+image,0)
        height = img.shape[0]
        width = img.shape[1]
        header = img[0:(int(height*0.33)), 0:width]
        footer = img[int(height*0.67):height, 0:width]
        left_body = img[int(height*0.33):int(height*0.67), 0:int(width*0.5)]
        right_body = img[int(height*0.33):int(height*0.67), int(width*0.5):width]
        header_path='train/header/'+folder+'/'+image.split('/')[-1]
        footer_path='train/footer/'+folder+'/'+image.split('/')[-1]
        left_path='train/left_body/'+folder+'/'+image.split('/')[-1]
        right_path='train/right_body/'+folder+'/'+image.split('/')[-1]
        if(os.path.exists(header_path)==False):
            cv2.imwrite(header_path, header)
        if(os.path.exists(footer_path)==False):
            cv2.imwrite(footer_path, footer)
        if(os.path.exists(left_path)==False):
            cv2.imwrite(left_path, left_body)
        if(os.path.exists(right_path)==False):
            cv2.imwrite(right_path, right_body)
        count+=1
        if(count==limit):
            break
			