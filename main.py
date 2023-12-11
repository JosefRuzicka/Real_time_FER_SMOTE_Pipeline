# Partially adapted from https://towardsdatascience.com/from-raw-images-to-real-time-predictions-with-deep-learning-ddbbda1be0e4
import pyrealsense2 as rs
import cv2 as cv
from model import FacialExpressionModel
import numpy as np

facec = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("VGG19_Josef_json.json", "VGG19_Josef.h5")
font = cv.FONT_HERSHEY_SIMPLEX

# CONNECT TO RS CAMERA
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# Try 1280, 720
#cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # DEPTH NO ES NECESARIA, IGUAL PUEDE OBSERVARSE POR GUSTO
pipe.start(cfg)
# RS CAMERA END

faceCascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # RS CAMERA
    frames = pipe.wait_for_frames()
    color_frame = frames.get_color_frame()
    
    #color_image = np.asanyarray(color_frame.get_data())
    fr = np.asanyarray(color_frame.get_data())
    # RS CAMERA END
    # ret, frame = cap.read()
    #cv.imshow("Captura", frame)

    gray_fr = cv.cvtColor(fr, cv.COLOR_BGR2GRAY)

    #JOSEF 2
    # Stack the grayscale image along the last axis to maintain the shape
    grayscale_image_with_channels = np.expand_dims(gray_fr, axis=-1)

    # Replicate the grayscale channel to match the expected shape (3 channels)
    frame = np.repeat(grayscale_image_with_channels, 3, axis=-1)

    faces = facec.detectMultiScale(frame, 1.3, 5)
        #print("faces", faces.__subclasshook__)

    for (x, y, w, h) in faces:
            #fc = gray_fr[y:y+h, x:x+w]
            fc = frame[y:y+h, x:x+w]
            #print("fc", fc.shape)
            roi = cv.resize(fc, (48, 48))
            #print("roi", roi.shape)
            #pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            pred = model.predict_emotion(roi[np.newaxis, :, :, :])

            cv.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv.rectangle(fr,(x,y),(x+w,y+h),(0,255,0),2)
            #cv.putText(frame, pred, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    #_, jpeg = cv.imencode('.jpg', frames)
    #jpeg.tobytes()

    if cv.waitKey(20) & 0xFF == ord('d'):
        pipe.stop()
        break
    
    cv.imshow("FER-CITIC", fr)

# RS CAMERA
#pipe.stop()
# RS CAMERA end