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

train_path = 'train'
test_path = 'test'
img_size = 48
batch_size = 64
epochs = 50

model = tf.keras.applications.VGG19(input_shape=(img_size,img_size,3),include_top=False,weights="imagenet")

# Adapt output to our classes
base_input = model.layers[0].input
base_output = model.layers[-2].output

final_output = layers.Dense(128)(base_output)
final_output = layers.Flatten()(final_output)
final_output = layers.Dropout(0.5)(final_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(7, activation='softmax')(final_output)
new_model = keras.Model(inputs= base_input, outputs= final_output)

new_model.compile(loss='categorical_crossentropy', 
                  optimizer = tf.keras.optimizers.SGD(learning_rate=0.0005),
                  metrics=['accuracy'],)

new_model.load_weights('VGG19_Josef.h5')

# serialize model to JSON
model_json = new_model.to_json()
with open("VGG19_Josef_json.json", "w") as json_file:
    json_file.write(model_json)

datagen_validation = ImageDataGenerator(horizontal_flip=True)
validation_generator = datagen_validation.flow_from_directory(test_path,
                                                    target_size=(img_size,img_size),
                                                    color_mode="rgb",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)

# Generate predictions for the validation set
validation_steps = validation_generator.n//validation_generator.batch_size+1
pred_labels = new_model.predict(validation_generator, steps=validation_steps)
pred_labels = np.argmax(pred_labels, axis=1)
true_labels = validation_generator.classes

# Print f1, precision, and recall scores
print("Precision: ", precision_score(true_labels, pred_labels , average="macro"))
print("Recall: ", recall_score(true_labels, pred_labels , average="macro"))
print("F1 Score: ", f1_score(true_labels, pred_labels , average="macro"))

# Generate confusion matrix
# using greens, blues and purples cmaps
cm = confusion_matrix(true_labels, pred_labels)
cmd = ConfusionMatrixDisplay.from_predictions(true_labels, pred_labels, display_labels=['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'], cmap='Greens')
