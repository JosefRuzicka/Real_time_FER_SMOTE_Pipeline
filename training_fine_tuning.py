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
train_sep_dir='smote_train/'

#####################
# Data Augmentation #
#####################
print('Augmenting Data')
datagen_train = ImageDataGenerator(horizontal_flip=True,
                                   zoom_range=0.1,
                                   width_shift_range = 0.1,
                                   height_shift_range = 0.1,
                                   rotation_range = 2.5)

train_generator = datagen_train.flow_from_directory(train_sep_dir,
                                                    target_size=(img_size,img_size),
                                                    color_mode="rgb",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

datagen_validation = ImageDataGenerator(horizontal_flip=True)
validation_generator = datagen_validation.flow_from_directory(test_path,
                                                    target_size=(img_size,img_size),
                                                    color_mode="rgb",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)

############################################
# Pre-Trained Model loading and adaptation #
############################################
with tf.device('/device:GPU:1'):
    print('Creating Model')
    # TODO: change this to an if at beginning of File.
    # For MobileNetV2 lr = 0.002. ResNet50 lr = 0.009, VGG19 -> 0.0005
    #model = tf.keras.applications.MobileNetV2(input_shape=(img_size,img_size,3),include_top=False,weights="imagenet") # pre trained model
    #model = tf.keras.applications.ResNet50(input_shape=(img_size,img_size,3),include_top=False,weights="imagenet")
    model = tf.keras.applications.VGG19(input_shape=(img_size,img_size,3),include_top=False,weights="imagenet")

    # Adapt output to our classes
    base_input = model.layers[0].input
    base_output = model.layers[-2].output

    final_output = layers.Dense(128)(base_output) # tried 128
    final_output = layers.Flatten()(final_output)
    final_output = layers.Dropout(0.5)(final_output)
    final_output = layers.Activation('relu')(final_output)
    final_output = layers.Dense(7, activation='softmax')(final_output)
    new_model = keras.Model(inputs= base_input, outputs= final_output)
    
    #####################
    # Transfer Learning #
    #####################
    print('Transfer-Learning')
    for layer in new_model.layers[:-6]: #-7 res net, mobileNet, -6 vgg,
        layer.trainable = False
    
    new_model.compile(loss='categorical_crossentropy', # tf.keras.losses.SparseCategoricalCrossentropy(), #[focal_loss(alpha=.25, gamma=3)],#keras_cv.losses.FocalLoss(), #loss=tf.keras.losses.SparseCategoricalCrossentropy()
                        optimizer=tf.keras.optimizers.SGD(learning_rate=0.0005),#, momentum=0.9, weight_decay=0.01),
                        metrics=['accuracy'],)
                        #callbacks=[metrics])

    #new_model.summary()

    steps_per_epoch = train_generator.n//train_generator.batch_size
    validation_steps = validation_generator.n//validation_generator.batch_size
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                patience=4, min_lr=0.000001, mode='auto') #4 1 0 menos
    checkpoint = ModelCheckpoint("VGG19_Josef_2.h5", monitor='val_accuracy',
                                save_weights_only=True, mode='max', verbose=1)
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
    callbacks = [checkpoint, reduce_lr, earlystopping]

    history = new_model.fit(
        x=train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data = validation_generator,
        validation_steps = validation_steps,
        callbacks=callbacks
    )

    ###############
    # Fine-Tuning #
    ###############
    print('Fine-Tuning')
    for layer in new_model.layers:  # Unfreeze the first 100 layers
        layer.trainable = True

    # use lr 0.00005 for VGG19, 0.0005 for the rest
    # Recompile the model with a lower learning rate
    #model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    new_model.compile(loss='categorical_crossentropy', # tf.keras.losses.SparseCategoricalCrossentropy(), #[focal_loss(alpha=.25, gamma=3)],#keras_cv.losses.FocalLoss(), #loss=tf.keras.losses.SparseCategoricalCrossentropy()
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),#, momentum=0.9, weight_decay=0.01),
                      metrics=['accuracy'])
    
    epochs = 50 
    steps_per_epoch = train_generator.n//train_generator.batch_size
    validation_steps = validation_generator.n//validation_generator.batch_size
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                patience=2, min_lr=0.000001, mode='auto')
    checkpoint = ModelCheckpoint("VGG19_Josef_2.h5", monitor='val_accuracy',
                                save_weights_only=True, mode='max', verbose=1)
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
    callbacks = [checkpoint, reduce_lr, earlystopping]

    history = new_model.fit(
        x=train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data = validation_generator,
        validation_steps = validation_steps,
        callbacks=callbacks
    )

    # serialize model to JSON
    model_json = new_model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)