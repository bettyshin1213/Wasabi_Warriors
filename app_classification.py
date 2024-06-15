import torch
import torch.nn as nn
from torchvision import models, transforms
import gradio as gr
from PIL import Image
import numpy as np

# 모델 로드
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# class_name = ['arctic_surf_clam', 'capelin_roe', 'crab_meat', 'fried_tofu_poouch', 'octopus', 'salmon', 'shrimp', 'tamagoyaki', 'tilapia', 'tuna']
class_name = ['arctic_surf_clam', 'capelin_roe', 'crab_meat', 'flatfish', 'fried_tofu_poouch', 'futomaki', 'octopus', 'salmon', 'shrimp', 'tamagoyaki', 'tilapia', 'tuna']
# model_path = 'models/sushi_classifier_resnet50_10.pth'  # 모델 파일 경로
model_path = 'models/sushiupdate_classifier_resnet50_10.pth'  # 모델 파일 경로

# ResNet50 모델 로드 및 수정
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_name))  # 클래스 개수에 맞게 수정

# 엄격하지 않은 로드
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()

# 이미지 전처리
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image):
    image = Image.fromarray(image)
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs[0], dim=0).cpu().numpy()
        print(outputs)
        print(probabilities)
    return {f'class_{i}({class_name[i]})': float(probabilities[i]) for i in range(len(probabilities))}

# Gradio 인터페이스 설정
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Webcam Image"),
    outputs=gr.Label(num_top_classes=10),
    live=True,
    title="Multi-Class Sushi Classification",
    description="Capture a sushi image from the webcam and classify it using CNN model."
)

if __name__ == "__main__":
    interface.launch(share=True)
