# Real_time_FER_SMOTE_Pipeline
My project for my computer science research class.

## Details
The FER_SMOTE_Pipeline.py runs the SMOTE, the training, the validation and saves the model.
This code is also modularized and can be run in separate steps by runnning: FER2013-SMOTE.py to create the balanced dataset from the original test folder, 
then training_fine_tuning.py to create, train and fine-tune the model. Optionally, validation.py can be run to test the trained model, 
and finally, main.py can be run to use the model with the realsense camera.

If you use the VGG19_Josef_json.json and the VGG19_Josef.h5 weights, you can skip all the previous steps and just run main.py 

## Requirements:
Intel Real Sense SKD 2.0. https://github.com/IntelRealSense/librealsense/releases
FER-2013 dataset (if you want to train a model again). https://www.kaggle.com/datasets/msambare/fer2013
