import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface_cov import RetinaFaceCoV
import timeit
import face_model
import argparse

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

#============================ Webcam 입력 ============================
cap = cv2.VideoCapture('test.webm')
cap.set(3, 1080) #WIDTH
cap.set(4, 640) #HEIGHT

font = cv2.FONT_HERSHEY_SIMPLEX
cnt = 1
#============================ 모델 실행 ============================
while True:
    cnt += 1
    scales = [640, 1080]
    # frame 별로 capture 한다
    ret, frame = cap.read()

    if ret is True:
        start_t = timeit.default_timer()
        if cnt % 5 ==0 :
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

# from __future__ import print_function
# import cv2
# import sys
# import numpy as np
# import datetime
# import os
# import glob
# from retinaface_cov import RetinaFaceCoV
# import timeit
# import face_model
# import argparse
# from tensorflow.python.keras.models import load_model
# from tensorflow.python.keras.preprocessing.image import img_to_array

# dir_path = os.getcwd()

# # ============================ Emotion 모델 불러오기 ============================
# classifier = load_model(dir_path + '/model/emotion_classification_vgg_5_emotions.h5')
# class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# # ============================ Mask 모델 불러오기 ============================
# thresh = 0.8
# mask_thresh = 0.2
# gpuid = 0

# detector = RetinaFaceCoV('./model/mnet_cov2', 0, gpuid, 'net3l')

# #============================ Age_Gender 모델 불러오기 ============================
# parser = argparse.ArgumentParser(description='face model test')
# # general
# parser.add_argument('--image-size', default='112,112', help='')
# # parser.add_argument('--image', default='Tom_Hanks_54745.png', help='')
# parser.add_argument('--model',
#                     default='model/model,0',
#                     help='path to load model.')
# parser.add_argument('--gpu', default=0, type=int, help='gpu id')
# parser.add_argument(
#     '--det',
#     default=0,
#     type=int,
#     help='mtcnn option, 1 means using R+O, 0 means detect from begining')
# args = parser.parse_args()

# model = face_model.FaceModel(args)
# gender = None

# #============================ Webcam 입력 ============================
# cap = cv2.VideoCapture('temp.mp4')
# cap.set(3, 1080) #WIDTH
# cap.set(4, 640) #HEIGHT

# font = cv2.FONT_HERSHEY_SIMPLEX
# cnt = 1
# #============================ 모델 실행 ============================
# while True:
#     cnt += 1
#     scales = [640, 1080]
#     # frame 별로 capture 한다
#     ret, frame = cap.read()
#     labels = []
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     if ret is True:
#         start_t = timeit.default_timer()
#         # ============================ Mask 착용 여부 확인 ============================
#         # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         im_shape = frame.shape
#         target_size = scales[0]
#         max_size = scales[1]
#         im_size_min = np.min(im_shape[0:2])
#         im_size_max = np.max(im_shape[0:2])

#         #if im_size_min>target_size or im_size_max>max_size:
#         im_scale = float(target_size) / float(im_size_min)
#         # prevent bigger axis from being more than max_size:
#         if np.round(im_scale * im_size_max) > max_size:
#             im_scale = float(max_size) / float(im_size_max)


#         scales = [im_scale]
#         flip = False
#         faces, landmarks = detector.detect(frame,
#                                             thresh,
#                                             scales=scales,
#                                             do_flip=flip)

#         if faces is not None:
#             print('find', faces.shape[0], 'faces')
#             for i in range(faces.shape[0]):
#                 # print('score', faces[i][4])
#                 face = faces[i]
#                 box = face[0:4].astype(np.int)
#                 mask = face[5]
                
#                 # print(i, box, mask)
#                 #color = (255,0,0)
#                 if mask >= mask_thresh:
#                     color = (0, 255, 0)
#                     text = 'with_mask'
#                 else:
#                     color = (0, 0, 255)
#                     text = 'without_mask'

#                     # ============================ Mask 미착용 시 나이와 성별 감지 ============================
#                     model_img = model.get_input(frame)

#                     if model_img is not None:
#                         gender, age = model.get_ga(model_img)
#                         if gender == 0:
#                             gender = 'female'
#                         else:
#                             gender = 'male'

#                         cv2.putText(frame, gender, (box[0], box[1] - 20), font, 0.7, color, 1)
#                         cv2.putText(frame, str(age), (box[0], box[1] - 35), font, 0.7, color, 1)

#                     roi_gray = gray[box[1]:box[3], box[0]:box[2]]
#                     roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
#                     if np.sum([roi_gray]) != 0:
#                         roi = roi_gray.astype('float') / 255.0
#                         roi = img_to_array(roi)
#                         roi = np.expand_dims(roi, axis=0)

#                         # make a prediction on the ROI, then lookup the class
#                         predicts = classifier.predict(roi)[0]
#                         label = class_labels[predicts.argmax()]
#                         label_position = (box[0], box[1])
#                         cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
#                     else:
#                         cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)


#                 cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
#                 cv2.putText(frame, text, (box[0], box[1] - 5), font, 0.7, color, 1)
#                 # cv2.putText(frame, gender, (box[0], box[1] - 20), font, 0.7, (0, 0, 0), 2)

#         terminate_t = timeit.default_timer()
#         FPS = int(1./(terminate_t - start_t))
#         print('FPS : ', FPS)
            
#         cv2.imshow('frame',frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()

