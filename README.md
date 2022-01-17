# Mask-Age-Gender-Emotion-estimation
### 소개

해당 데모 프로그램은 사용자가 마스크를 착용 했는지 감별하고, 마스크 미착용자에 대해 사용자의 성별, 나이, 감정을 예측하여 화면에 출력하는 프로그램입니다.
### 1. 사용 모델

총 **4개의 모델** 사용

1. mobilenet (based on mxnet) **mask on/off** detecting [deepinsight/insightface](https://github.com/deepinsight/insightface/tree/master/detection/retinaface_anticov)
 2. SSR-net (based on mxnet) **gender** estimation [wayen820/gender_age_estimation_mxnet](https://github.com/wayen820/gender_age_estimation_mxnet)
 3. SSR-net (based on mxnet) **age** estimation [wayen820/gender_age_estimation_mxnet](https://github.com/wayen820/gender_age_estimation_mxnet)
 4.  vgg (based on tensorflow)- 5 **emotions** estimation (화남, 행복, 무표정, 슬픔, 놀람) [rohwid/emotion-recognition](https://github.com/rohwid/emotion-recognition)


### 2.  실행 방법
1. **환경 설정**
    - 가상환경 생성
    
    ```python
    $ conda create -n mask_demo python=3.8
    $ conda activate mask_demo 
    ```
    
    - cuda 10.1
    - cudnn 7.6.5
    - tensorflow 2.2
    - python 3.8
    - mxnet-cu101

2. **실행**

    person.py
    - 마스크 쓴 사람 중 가장 가까이 있는 사람의 성별, 나이, 감정을 예측
    
    people.py
    - 마스크를 쓰지 않은 사람 **모두에 대해** 성별, 나이, 감정을 예측
    - 처리 속도 향상을 위해 거리가 기준보다 가까운 사람의 성별, 나이, 감정만 예측
    
    ```python
    # Webcam ver 
    # 한사람만
    $ python person.py
    
    # 여러명 탐지시 
    $ python people.py
    ```
    
    ** 동영상에서 사용시 
    
    ```python
    def main(args):
    	cap=cv2.VideoCapture('video/test.webm')  ## 주석해제
    	# cap=cv2.VideoCapture(0) ## 주석처리
    ```
    
### 3.  애로사항

- 웹캠 상에서 4개의 모델을 동시에 모두 사용하면 fps가 낮아져 실시간 Demo 불가능 
→ 빠른 업데이트가 필요한 필요한 마스크 착용 구분, 감정 인식은 실시간으로 처리하고 **나이, 성별은 n frame 당 한번씩 예측하고 업데이트** 되도록 알고리즘을 구현하여 fps 개선
- people.py의 경우 20명 이상 사람이 사용할 경우 처리 속도가 많이 느려짐
- 감정 인식시 '화남' class 가 잘 나오지 않음
    
    → 학습 데이터 편향으로 인한 문제 (서양인 데이터로 측면, 찡그리며 입벌린 데이터가 다수)
    → Asian emotion train data를 새로 받아 학습한다면 개선 가능
