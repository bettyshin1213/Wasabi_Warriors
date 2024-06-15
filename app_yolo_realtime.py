import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np

# YOLOv5 모델 로드
# model = torch.hub.load('yolov5/runs/exp8/weights/best91.pt', 'yolov5m')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolov5/yolov5m_adamw_cossine_91.pt', force_reload=True)


# Streamlit 설정
st.title('Real-Time Sushi Object Detection with YOLOv5')
st.sidebar.title('Settings')
st.sidebar.subheader('Webcam Settings')

# 웹캠 설정
use_webcam = st.sidebar.checkbox('Use Webcam', value=True)
confidence = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.5)
st.sidebar.markdown('---')

# 비디오 스트림 열기
if use_webcam:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture('path/to/video')

stframe = st.empty()

# 비디오 스트리밍
while True:
    ret, frame = cap.read()
    if not ret:
        st.write("웹캠을 열 수 없습니다.")
        break

    # YOLOv5 모델로 객체 감지
    results = model(frame)

    # 감지 결과를 프레임에 그리기
    frame = np.squeeze(results.render())

    # Streamlit에 프레임 표시
    stframe.image(frame, channels='BGR')

    # 종료 조건 (예: 'q' 키를 누르면 종료)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
