import mxnet as mx
import cv2
import sys
import os
import numpy as np
import align_tools
import face_model
import detect
from retinaface_cov import RetinaFaceCoV
from tensorflow.python.keras.models import load_model
import timeit
from PIL import ImageFont, ImageDraw, Image

'''
this file is used for test slim model
'''
model_str='./model/ssr2_megaage_1_1/model,0'
model_gender_str='./model/ssr2_imdb_gender_1_1/model,0'
gpu=0

fontpath_s = "fonts/AppleSDGothicNeoR.ttf"
fontpath_l = "fonts/AppleSDGothicNeoH.ttf"


def get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym,data_names=('data','stage_num0','stage_num1','stage_num2'),context=ctx, label_names = None)
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1])),('stage_num0',(1,3)),('stage_num1',(1,3)),('stage_num2',(1,3))])
  model.set_params(arg_params, aux_params)
  return model

def get_model_gender(ctx, image_size, model_str, layer):
  _vec = model_gender_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym,data_names=('data','stage_num0','stage_num1','stage_num2'),context=ctx, label_names = None)
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1])),('stage_num0',(1,3)),('stage_num1',(1,3)),('stage_num2',(1,3))])
  model.set_params(arg_params, aux_params)
  return model

def main(args):
    align_t=align_tools.align_tools()
    cap=cv2.VideoCapture('test.webm')
    # cap=cv2.VideoCapture(0)
    count = 1
    cap.set(3,1920) #WIDTH
    cap.set(4, 1080) #HEIGHT
    font = cv2.FONT_HERSHEY_SIMPLEX
    age = [0]
    g = None
    while cap.isOpened():
        scales = [640, 1080]
        ret,frame=cap.read()
        start_t = timeit.default_timer()
        if ret is True:
            faces, person_num = detect.detect_person(frame, scales, thresh, detector)
            if faces is not None:
                for i in range(person_num):
                    box, color, text, mask_on = detect.detect_mask(faces, i, mask_thresh)
                    w = int(box[2] - box[0])
                    d = int((box[2] - box[0])/2)
                    font_size = int((box[2] - box[0])/7)
                    font_size_cv = d/150
                    font_ko_s = ImageFont.truetype(fontpath_s, font_size)
                    font_ko_l = ImageFont.truetype(fontpath_l, font_size)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(frame, text, (box[0], box[1] + font_size ), font, font_size_cv, color, 1)
                    box_sizee = (box[2]-box[0]) * (box[3]-box[1])
                    p_size = (box_sizee / (scales[0]*scales[1]))*100
                        # print('d: %d'%d)
                    if (mask_on is False) & (p_size > 0.3):
                        p_size = 0
                        label = detect.detect_emotion(frame, box, classifier, font)
                        # if (count % 5 == 0) or (count == 1):
                        detect.detect_age_gender_people(frame, box, model_age, model_gender, align_t, font)


            cv2.imshow('frame',frame)
            count += 1
            terminate_t = timeit.default_timer()
            FPS = int(1./(terminate_t - start_t))
            print('FPS : ', FPS)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = detect.get_args()
    thresh = 0.9
    mask_thresh = 0.2
    gpuid = 0
    model = face_model.FaceModel(args)
    detector = RetinaFaceCoV('./model/mnet_cov2', 0, gpuid, 'net3l')
    model_age =get_model(mx.gpu(gpu),(64,64),model_str,'_mulscalar16')
    model_gender=get_model_gender(mx.gpu(gpu),(64,64),model_gender_str,'_mulscalar16')
    classifier = load_model('./model/emotion_classification_vgg_5_emotions.h5')
    main(args)
