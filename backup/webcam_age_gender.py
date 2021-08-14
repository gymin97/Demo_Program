import cv2
import sys
import face_model
import argparse
import numpy as np
import datetime
import timeit
import detect

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
font = cv2.FONT_HERSHEY_SIMPLEX

# 웹캠에서 영상 읽어오기
cap = cv2.VideoCapture(0)
cap.set(3, 1080) #WIDTH
cap.set(4, 640) #HEIGHT

while True:
    ret, img = cap.read()

    if ret is True:
        start_t = timeit.default_timer()
        
        model_img = model.get_input(img)
        if model_img is not None:
            gender, age = model.get_ga(model_img)
            if gender == 0:
                gender = 'female'
            else:
                gender = 'male'
        
            terminate_t = timeit.default_timer()
            FPS = int(1./(terminate_t - start_t))
            print('FPS : ', FPS)
                

            # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(img, gender, (10, 10), font, 0.7, (0, 0, 0), 1)

        cv2.imshow('frame',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
