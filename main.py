import os
import cv2 # type: ignore
import torch # type: ignore
import torchvision.models as models # type: ignore
import torchvision.transforms as transforms # type: ignore
from PIL import Image # type: ignore
from ultralytics import YOLO # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore

# Function to check YOLO setup
def check_yolo_setup():
    from IPython.display import display, Image as IPImage
    import ultralytics
    ultralytics.checks()
    print("Yolo setup is good to go!")

# Function to initialize YOLO model
def initialize_yolo(model_path='yolov8n.pt'):
    return YOLO(model_path)

# Function to extract features using a pre-trained ResNet model
def extract_features(image_path, model, preprocess):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        features = model(image)

    return features.squeeze().numpy()

check_yolo_setup()
yolo_model = initialize_yolo()
resnet_model = models.resnet50(pretrained=True)


