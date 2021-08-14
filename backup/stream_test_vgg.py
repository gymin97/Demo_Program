from __future__ import print_function

import os
import cv2
import numpy as np

from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import img_to_array

dir_path = os.getcwd()

face_classifier = cv2.CascadeClassifier(dir_path + '/model/haarcascade_frontalface_default.xml')
classifier = load_model(dir_path + '/model/emotion_classification_vgg_5_emotions.h5')

class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture('temp.mp4')

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces3 = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces3:
        # cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # roi_gray = gray[box[1]:box[3], box[0]:box[2]]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        # rect,face,image = face_detector(frame)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # make a prediction on the ROI, then lookup the class
            predicts = classifier.predict(roi)[0]
            label = class_labels[predicts.argmax()]
            label_position = (box[0], box[1])
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
