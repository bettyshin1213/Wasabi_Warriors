import torch
import cv2
import numpy as np
import gradio as gr

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

#model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolov5/yolov5m_adamw_cossine_91.pt', force_reload=True)
model.eval()

# 추론 함수
def detect_objects(image):
    # 이미지를 YOLOv5 모델의 입력 형식으로 변환
    results = model(image)
    
    # 결과를 처리하여 반환
    detections = results.pandas().xyxy[0].to_dict(orient="records")
    annotated_image = np.array(image)
    for det in detections:
        cv2.rectangle(annotated_image, 
                      (int(det['xmin']), int(det['ymin'])), 
                      (int(det['xmax']), int(det['ymax'])), 
                      (0, 255, 0), 2)
        cv2.putText(annotated_image, 
                    det['name'], 
                    (int(det['xmin']), int(det['ymin']) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 255, 0), 2)
    return annotated_image

# Gradio 인터페이스 설정
interface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="numpy", label="Input Image"),
    outputs=gr.Image(type="numpy", label="Detected Objects"),
    live=True,
    title="YOLOv5 Sishi Object Detection",
    description="Upload a sushi image or use your webcam to detect sushi using YOLOv5."
)

# Gradio 인터페이스 실행
interface.launch(share=True)
