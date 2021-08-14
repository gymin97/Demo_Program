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
import datetime

'''
this file is used for test slim model
'''
model_str='./model/ssr2_megaage_1_1/model,0'
model_gender_str='./model/ssr2_imdb_gender_1_1/model,0'
gpu=0

fontpath_s = "fonts/AppleSDGothicNeoR.ttf"
fontpath_l = "fonts/AppleSDGothicNeoH.ttf"


def cutiing_box(frame,box):
    x0,y0,x1,y1 = box
    if box[0]-0.2*box[0] > 0:
        x0 = box[0]-0.2*box[0]
    if box[1] - 0.2*box[1] > 0:
        y0 = box[1] - 0.2*box[1]

    if box[2] + 0.2*box[2] < frame.shape[0]:
        x1 = box[2] + 0.2*box[2]

    if box[3] + 0.2*box[3] < frame.shape[1]:
        y1 = box[3] + 0.2*box[3]
    return frame[int(y0):int(y1),int(x0):int(x1),:]




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
    #cap=cv2.VideoCapture('video/test.webm')
    cap=cv2.VideoCapture(0)
    count = 1
    # width, height = 640,480
    # width,height = 1280, 720
    width,height = 1920, 1080
    #width,height = 1520, 1200
    cap.set(3, width) #WIDTH
    cap.set(4, height) #HEIGHT
    font = cv2.FONT_HERSHEY_SIMPLEX
    age = [0]
    g = None
    mask_on = None
    CI_r = cv2.imread('fonts/piai_CI.png')
    CI = cv2.resize(CI_r, (0, 0), fx=0.3, fy=0.3)
    CI_w, CI_h, _ = CI.shape
    cnt = 0
    font_ko_o = ImageFont.truetype(fontpath_l, 30)

    while cap.isOpened():
        scales = [640, 1080]
        ret,frame=cap.read()
        frame = cv2.flip(frame, 1)        
        ## PIAI 로고삽입 
        cv2.rectangle(frame, (0, frame.shape[0]- CI_w), (frame.shape[1] , frame.shape[0]), (255,255,255), -1)  
        frame[frame.shape[0] - CI_w : frame.shape[0], frame.shape[1] - CI_h : frame.shape[1]] = CI

        start_t = timeit.default_timer()
        if ret is True:
            faces, person_num = detect.detect_person(frame, scales, thresh, detector)
            if faces is not None:
                box_list = []
                size_list= []
                mask_F=0
                for i in range(person_num):
                    box, color, text, mask_on = detect.detect_mask(faces, i, mask_thresh)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

                    box_sizee = (box[2]-box[0]) * (box[3]-box[1])
                    p_size = (box_sizee / (scales[0]*scales[1]))*100

                    if mask_on == False:
                        box_list.append(list(box[:]))
                        size_list.append(p_size)
                        mask_F = 1
                    
                    w = int(box[2] - box[0])
                    d = int((box[2] - box[0])/2)
                    font_size = int((box[2] - box[0])/6)
                    font_ko_s = ImageFont.truetype(fontpath_s, font_size)
                    font_ko_l = ImageFont.truetype(fontpath_l, font_size)
                    frame = Image.fromarray(frame)
                    draw = ImageDraw.Draw(frame)
                    draw.text((box[0]+font_size/2 , box[1] + font_size/4 ),'%s'%text, font = font_ko_s, fill=color)
                    frame = np.array(frame)

                if (mask_F == 1) and (p_size>0.4):
                    max_id = np.argmax(size_list)
                    box = box_list[max_id]
                    p_size = 0
                    label = detect.detect_emotion(frame, box, classifier, font)
                    crop_frame = cutiing_box(frame,box)
                    if (count % 20 == 0) or (count == 1):
                        age, g = detect.detect_age_gender_ko(crop_frame, box, model_age, model_gender, align_t)
                        if int(age[0])<21:
                            age[0] = 21
                        elif int(age[0])>60:
                            age[0]=60
                        if g == None:
                            g='남성'
                        

                    # print ('count, box', count, box)
                    font_size = int((box[2] - box[0])/6)
                    font_ko_s = ImageFont.truetype(fontpath_s, font_size)
                    font_ko_l = ImageFont.truetype(fontpath_l, font_size)
                    frame = Image.fromarray(frame)
                    draw = ImageDraw.Draw(frame)
                    text = str(int(age[0])+3) + '세 ' + str(g)                   
                    draw.text((box[0] + 5, box[3] + 10 ), text , font = font_ko_l, fill=(0, 0, 0))
                    draw.text((box[0] + 5, box[3] + 10 ), text , font = font_ko_s, fill=(255, 255, 255)) 
                    frame = np.array(frame)


            count += 1
            terminate_t = timeit.default_timer()
            FPS = int(1./(terminate_t - start_t))
            print('FPS : ', FPS)

            now = datetime.datetime.now().strftime("%d_%H-%M-%S")
            keycode = cv2.waitKey(1)
            #frame = cv2.resize(frame, dsize=(1280, 1024), interpolation=cv2.INTER_AREA)
            w, h = frame.shape[0], frame.shape[1]

            if keycode == ord('q') or keycode == ord('Q'):
                break

            elif keycode ==ord('s') and cnt ==0 :
                cv2.rectangle(frame, (0, 0), (1920, 1080), (255, 255, 255), -1)
                cv2.imshow('frame', frame)
                cnt+=1

            elif keycode ==ord('s') or (cnt >0 and cnt <10):
                cv2.imwrite('screenshot/' + str(now) + ".png", frame)
                text= 'screenshot saved'
                frame = Image.fromarray(frame)
                draw = ImageDraw.Draw(frame)
                draw.text((w*0.5, h*0.71),  text, font=font_ko_o, fill=(0,0,0))
                frame = np.array(frame)
                cv2.imshow('frame', frame)
                cnt+=1

            elif cnt >=10:
                cnt = 0
                cv2.imshow('frame', frame)
            else:
                text = 'press "s" to save screenshot'
                frame = Image.fromarray(frame)
                draw = ImageDraw.Draw(frame)
                draw.text((w*0.5, h*0.71),  text, font=font_ko_o, fill=(0,0,0))
                frame = np.array(frame)
                cv2.imshow('frame', frame)

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
