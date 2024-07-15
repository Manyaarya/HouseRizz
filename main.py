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

# Function to perform object detection and save cropped images
def detect_and_crop_objects(model, image_url, cropped_images_dir='cropped_images', conf=0.6):
    results = model(source=image_url, conf=conf)
    os.makedirs(cropped_images_dir, exist_ok=True)

    for i, result in enumerate(results):
        img = result.orig_img
        for j, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_img = img[y1:y2, x1:x2]
            cropped_img_path = os.path.join(cropped_images_dir, f"cropped_{i}_{j}.jpg")
            cv2.imwrite(cropped_img_path, cropped_img)
            print(f"Cropped image saved to {cropped_img_path}")
    return results

check_yolo_setup()
yolo_model = initialize_yolo()
resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Perform object detection and crop objects
results = detect_and_crop_objects(yolo_model, "https://i.pinimg.com/564x/63/06/6b/63066b20ad95c537334e9e4885bc07f2.jpg")



