import os
import sys
from mtcnn_detector import MtcnnDetector
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import face_preprocess
import mxnet as mx
import cv2
import numpy as np


class align_tools:
    def __init__(self,det=0,image_size=(64,64)):
        '''
        얼굴 정렬 도구
        : param det : 0 모든 감지, 1 키 포인트 감지 및 정렬, 잘린 얼굴 이미지에 사용
        : param image_size :
        '''
        self.det=det
        self.image_size=image_size
        self.ctx=mx.gpu(0)
        det_threshold = [0.6, 0.7, 0.8]
        mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
        if det == 0:
            self.detector = MtcnnDetector(model_folder=mtcnn_path, ctx=self.ctx, num_worker=1, accurate_landmark=True,
                                     threshold=det_threshold)
        else:
            self.detector = MtcnnDetector(model_folder=mtcnn_path, ctx=self.ctx, num_worker=1, accurate_landmark=True,
                                     threshold=[0.0, 0.0, 0.2])

    # 원본 input
    def get_intput_cv(self,face_img):
        ret = self.detector.detect_face(face_img, det_type=self.det)
        if ret is None:
            return None,None
        bounding_boxes, points = ret
        # print('bb', bounding_boxes)
        if bounding_boxes.shape[0] == 0:
            return None,None
        nrof_faces = bounding_boxes.shape[0]
        det = bounding_boxes[:, 0:4]
        img_size = np.asarray(face_img.shape)[0:2]
        bindex = 0
        if nrof_faces > 1:
            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img_size / 2
            offsets = np.vstack(
                [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            bindex = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
        _bbox = bounding_boxes[bindex, 0:4]
        _landmark = points[bindex, :].reshape((2, 5)).T
        warped = face_preprocess.preprocess(face_img, bbox=_bbox, landmark=_landmark)
        warped = cv2.resize(warped, self.image_size)
        return warped,_bbox


    def get_intput_cv_people(self,frame, model_age, model_gender, box, font):
        w = int(box[2] - box[0])
        d = int((box[2] - box[0])/2)
        font_size = int((box[2] - box[0])/7)
        font_size_cv = d/150
        if font_size_cv < 0.6 :
            font_size_cv = 0.6
        ret = self.detector.detect_face(frame, det_type=self.det)
        if ret is None:
            return None,None
        bounding_boxes, points = ret
        # print(bounding_boxes, points)
        if bounding_boxes.shape[0] == 0:
            return None,None
        nrof_faces = bounding_boxes.shape[0]
        det = bounding_boxes[:, 0:4]
        img_size = np.asarray(frame.shape)[0:2]
        bindex = 0
        for bindex in range(nrof_faces):
            _bbox = bounding_boxes[bindex, 0:4]
            _landmark = points[bindex, :].reshape((2, 5)).T
            warped = face_preprocess.preprocess(frame, bbox=_bbox, landmark=_landmark)
            nimg = cv2.resize(warped, self.image_size)
            if nimg is None:
                return [0], None
            nimg = nimg[:, :, ::-1]
            nimg= np.transpose(nimg,(2,0,1))

            input_blob = np.expand_dims(nimg, axis=0)
            data = mx.nd.array(input_blob)
            db = mx.io.DataBatch(data=(data,mx.nd.array([[0,1,2]]),mx.nd.array([[0,1,2]]),mx.nd.array([[0,1,2]])))
            
            model_age.forward(db, is_train=False)
            age = model_age.get_outputs()[0].asnumpy()

            model_gender.forward(db,is_train=False)
            gender=model_gender.get_outputs()[0].asnumpy()

            if gender[0]>0.5:
                g='male'
            else: g='female'

        cv2.putText(frame, '%d'%age, (box[2] + 10, box[1] + 20), font, font_size_cv, (0, 0, 0), 5)
        cv2.putText(frame, '%d'%age, (box[2] + 10, box[1] + 20), font, font_size_cv, (255, 255, 255), 2)
        cv2.putText(frame, '%s'%g, (box[2] + 10, box[1] + 20 + font_size), font, font_size_cv, (0, 0, 0), 5)
        cv2.putText(frame, '%s'%g, (box[2] + 10, box[1] + 20 + font_size), font, font_size_cv, (255, 255, 255), 2)
        return None, None

    def get_input(self, img_file,clear=False):
        '''
        정렬 된 이미지를 반환하고 HWC, bgr 형식을 반환합니다. 이미지에 여러 개의 얼굴이있는 경우 이미지 중간에 있고 크기가 더 큰 얼굴을 우선적으로 선택합니다.
        : param img_file : 이미지 경로, clear : imagesize보다 작은 얼굴 크기 지우기
        : return : 얼굴 이미지 정렬, HWC bgr 형식
        '''
        face_img=cv2.imread(img_file)
        ret = self.detector.detect_face(face_img, det_type=self.det)
        if ret is None:
            return None
        bounding_boxes, points = ret
        if bounding_boxes.shape[0] == 0:
            return None
        nrof_faces = bounding_boxes.shape[0]
        det = bounding_boxes[:, 0:4]
        img_size = np.asarray(face_img.shape)[0:2]
        bindex = 0
        if nrof_faces > 1:
            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img_size / 2
            offsets = np.vstack(
                [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            bindex = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
        _bbox = bounding_boxes[bindex, 0:4]
        _bbox_size=(_bbox[2]-_bbox[0])*(_bbox[3]-_bbox[1])
        if clear and _bbox_size<self.image_size[0]*self.image_size[1]:
            return None
        _landmark = points[bindex,:].reshape((2, 5)).T
        warped = face_preprocess.preprocess(face_img, bbox=_bbox, landmark=_landmark)
        warped= cv2.resize(warped,self.image_size)
        return warped




