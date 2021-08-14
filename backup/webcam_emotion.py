from pathlib import Path
import cv2
import dlib
import sys
import numpy as np
import argparse
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from os import listdir
from os.path import isfile, join
import datetime
import os
import glob
from retinaface_cov import RetinaFaceCoV
import timeit
import face_model

classifier = load_model('./model/emotion_little_vgg_2.h5')
emotion_classes = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}


# ============================ Mask 모델 불러오기 ============================
thresh = 0.8
mask_thresh = 0.2
gpuid = 0

detector = RetinaFaceCoV('./model/mnet_cov2', 0, gpuid, 'net3l')

#============================ Age_Gender 모델 불러오기 ============================
parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
# parser.add_argument('--image', default='Tom_Hanks_54745.png', help='')
parser.add_argument('--model',
                    default='model/model,0',
                    help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument(
    '--det',
    default=0,
    type=int,
    help='mtcnn option, 1 means using R+O, 0 means detect from begining')
args = parser.parse_args()

model = face_model.FaceModel(args)
gender = None



def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

margin = 0.4
img_size = 64
detector2 = dlib.get_frontal_face_detector()

# Initialize Webcam
cap = cv2.VideoCapture('temp.mp4')
cap.set(3, 1080) #WIDTH
cap.set(4, 640) #HEIGHT

font = cv2.FONT_HERSHEY_SIMPLEX
cnt = 1
while True:
    scales = [640, 1080]
    ret, frame = cap.read()
    preprocessed_faces_emo = []           
 
    input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = np.shape(input_img)
    detected = detector2(frame, 1)
    faces = np.empty((len(detected), img_size, img_size, 3))

    
    if ret is True:
        start_t = timeit.default_timer()
        # ============================ Mask 착용 여부 확인 ============================
        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im_shape = frame.shape
        target_size = scales[0]
        max_size = scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        #if im_size_min>target_size or im_size_max>max_size:
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)


        scales = [im_scale]
        flip = False
        faces, landmarks = detector.detect(frame,
                                            thresh,
                                            scales=scales,
                                            do_flip=flip)

        if faces is not None:
            print('find', faces.shape[0], 'faces')
            for i in range(faces.shape[0]):
                # print('score', faces[i][4])
                face = faces[i]
                box = face[0:4].astype(np.int)
                mask = face[5]
                
                # print(i, box, mask)
                #color = (255,0,0)
                if mask >= mask_thresh:
                    color = (0, 255, 0)
                    text = 'with_mask'
                else:
                    color = (0, 0, 255)
                    text = 'without_mask'

                    # ============================ Mask 미착용 시 나이와 성별 감지 ============================
                    model_img = model.get_input(frame)

                    if model_img is not None:
                        gender, age = model.get_ga(model_img)
                        if gender == 0:
                            gender = 'female'
                        else:
                            gender = 'male'

                        cv2.putText(frame, gender, (box[0], box[1] - 20), font, 0.7, color, 1)
                        cv2.putText(frame, str(age), (box[0], box[1] - 35), font, 0.7, color, 1)

                    if len(detected) > 0:
                        for i, d in enumerate(detected):
                            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                            xw1 = max(int(x1 - margin * w), 0)
                            yw1 = max(int(y1 - margin * h), 0)
                            xw2 = min(int(x2 + margin * w), img_w - 1)
                            yw2 = min(int(y2 + margin * h), img_h - 1)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                            faces[i, :, :, :] = cv2.resize(frame[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                            face =  frame[yw1:yw2 + 1, xw1:xw2 + 1, :]
                            face_gray_emo = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                            face_gray_emo = cv2.resize(face_gray_emo, (48, 48), interpolation = cv2.INTER_AREA)
                            face_gray_emo = face_gray_emo.astype("float") / 255.0
                            face_gray_emo = img_to_array(face_gray_emo)
                            face_gray_emo = np.expand_dims(face_gray_emo, axis=0)
                            preprocessed_faces_emo.append(face_gray_emo)

                        # make a prediction for Emotion 
                        emo_labels = []
                        for i, d in enumerate(detected):
                            preds = classifier.predict(preprocessed_faces_emo[i])[0]
                            emo_labels.append(emotion_classes[preds.argmax()])
                        
                        # draw results
                        for i, d in enumerate(detected):
                            label = "{}".format(emo_labels[i])
                            draw_label(frame, (d.left(), d.top()), label)

                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, text, (box[0], box[1] - 5), font, 0.7, color, 1)
                # cv2.putText(frame, gender, (box[0], box[1] - 20), font, 0.7, (0, 0, 0), 2)

        terminate_t = timeit.default_timer()
        FPS = int(1./(terminate_t - start_t))
        print('FPS : ', FPS)
            
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


