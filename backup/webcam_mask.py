import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface_cov import RetinaFaceCoV
import timeit

# 모델 불러오기
thresh = 0.8
mask_thresh = 0.2

count = 1

gpuid = 0
#detector = RetinaFaceCoV('./model/mnet_cov1', 0, gpuid, 'net3')
detector = RetinaFaceCoV('./model/mnet_cov2', 0, gpuid, 'net3l')

# 웹캠에서 영상 읽어오기
cap = cv2.VideoCapture('mask.mp4')
cap.set(3, 1080) #WIDTH
cap.set(4, 640) #HEIGHT

prev_time = 0 # FPS 계산
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    scales = [640, 1080]
    # frame 별로 capture 한다
    ret, img = cap.read()

    if ret is True:
        start_t = timeit.default_timer()

        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im_shape = img.shape
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

        for c in range(count):
            faces, landmarks = detector.detect(img,
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
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(img, text, (box[0], box[1] - 5), font, 0.7, color, 1)


        terminate_t = timeit.default_timer()
        FPS = int(1./(terminate_t - start_t))
        print('FPS : ', FPS)
            
        cv2.imshow('frame',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

