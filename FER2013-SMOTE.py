import pandas as pd
import numpy as np
import os
import sys
import shutil
from shutil import copyfile
import os.path
import cv2 as cv
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array,load_img
from PIL import Image
from sklearn.model_selection import train_test_split
from numpy import load
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

img_size = 48
batch_size = 64
epochs = 50

imagegen = ImageDataGenerator()

train_path = 'train'
test_path = 'test'

# TODO cambiar 48 por image size
print('Generating train generator')
train_generator = imagegen.flow_from_directory(train_path, class_mode="categorical", shuffle=False, batch_size=128, target_size=(48, 48),seed=42)

x = np.concatenate([train_generator.next()[0] for i in range(train_generator.__len__())])
y = np.concatenate([train_generator.next()[1] for i in range(train_generator.__len__())])

# #Converting  our color images to a vector
# image count, img size, img size, color layers.
X_train = x.reshape(28709, 48*48*3) 

###########################
# Oversampling with SMOTE #
###########################
print('Oversampling with SMOTE')
sm = SMOTE(random_state=2)
X_smote, y_smote = sm.fit_resample(X_train, y)
Xsmote_img=X_smote.reshape(7215*7,48,48,3) # (Max(images_per_class)) * classes

#Save all images generated by the SMOTE method to the drive
train_sep_dir='smote_train/'

#Create a "testfolder" if it does not exist on the drive
if not os.path.exists(train_sep_dir):
    os.mkdir(train_sep_dir)

categories = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}

#########################
# SMOTE Images Labeling #
#########################
print('Labeling SMOTE Images')
def get_key(val):
    for key, value in categories.items():
         if val[0] == value:
             #print(key)
             return key

for i in range(len(Xsmote_img)):
  #label=get_key(str(y_smote[i]))
  label=get_key(np.where(y_smote[i]==1))
  #print((np.where(y_smote[i]==1)))
  if not os.path.exists(train_sep_dir + '/' + str(label)):
    os.mkdir(train_sep_dir + str(label))
  pil_img = array_to_img(Xsmote_img[i]* 255)
  pil_img.save(train_sep_dir + str(label) +'/'+ 'smote_'+ str(i) + '.jpg')