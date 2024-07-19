import os
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to initialize YOLO model
def initialize_yolo(model_path='yolov8n.pt'):
    return YOLO(model_path)

# Function to extract features using a pre-trained ResNet model
def extract_features(image, model, preprocess):
    image = preprocess(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        features = model(image)

    return features.squeeze().numpy()

# Function to clear the cropped_images directory
def clear_directory(directory):
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)

# Function to perform object detection and save cropped images
def detect_and_crop_objects(model, image, cropped_images_dir='cropped_images', conf=0.6):

    clear_directory(cropped_images_dir)
    results = model(source=image, conf=conf)
    os.makedirs(cropped_images_dir, exist_ok=True)

    cropped_count = 0
    detected_categories = set()
    for i, result in enumerate(results):
        img = result.orig_img
        for j, box in enumerate(result.boxes):
            category = result.names[int(box.cls[0])]
            detected_categories.add(category)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_img = img[y1:y2, x1:x2]
            cropped_img_path = os.path.join(cropped_images_dir, f"cropped_{i}_{j}.jpg")
            cv2.imwrite(cropped_img_path, cropped_img)
            cropped_count += 1
            logger.info(f"Cropped image saved to {cropped_img_path}")
    logger.info(f"Detected categories: {detected_categories}")
    return cropped_count, detected_categories

# Function to fetch images from local directory
def fetch_images_from_local(directory='images', categories=None):
    images = {}
    if categories is None:
        categories = os.listdir(directory)
    
    for category in categories:
        category_dir = os.path.join(directory, category)
        if os.path.isdir(category_dir):
            for img_file in os.listdir(category_dir):
                img_path = os.path.join(category_dir, img_file)
                if os.path.isfile(img_path):
                    images[os.path.join(category, img_file)] = Image.open(img_path).convert('RGB')
    return images

# Function to extract features from cropped images
def extract_features_from_cropped_images(cropped_images_dir, model, preprocess):
    feature_dict = {}
    for img_path in os.listdir(cropped_images_dir):
        full_img_path = os.path.join(cropped_images_dir, img_path)
        if os.path.isfile(full_img_path):
            image = Image.open(full_img_path).convert('RGB')
            features = extract_features(image, model, preprocess)
            feature_dict[img_path] = features
    logger.info("Features extracted for all cropped images.")
    return feature_dict

# Function to extract features from catalog images within specific categories
def extract_features_from_catalog_images(catalog_images, model, preprocess, categories):
    catalog_features = {}
    for category in categories:
        for img_path, image in catalog_images.items():
            if category in img_path:
                features = extract_features(image, model, preprocess)
                catalog_features[img_path] = features
                logger.info(f"Features extracted for {img_path}")
    logger.info("Features extracted for all catalog images in detected categories.")
    return catalog_features

# Function to find similar items
def find_similar_items(features, catalog_features, top_k=5):
    similarities = []
    for img_path, catalog_feature in catalog_features.items():
        similarity = cosine_similarity(features.reshape(1, -1), catalog_feature.reshape(1, -1))[0][0]
        similarities.append((img_path, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Function to generate recommendations
def generate_recommendations(feature_dict, catalog_features):
    recommendations = {}
    for img_name, features in feature_dict.items():
        similar_items = find_similar_items(features, catalog_features)
        recommendations[img_name] = [(item_img, float(similarity)) for item_img, similarity in similar_items]
    logger.info("Recommendations generated.")
    return recommendations

# Initialize models and preprocess
yolo_model = initialize_yolo()
resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    logger.info("Root endpoint called")
    return {"message": "Welcome to the Similarity API"}

@app.post("/recommend")
async def recommend(file: UploadFile = File(...)):
    logger.info("Recommend endpoint called")

    # Load the uploaded image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Save the uploaded image temporarily
    temp_image_path = "temp_image.jpg"
    image.save(temp_image_path)

    # Perform object detection and cropping
    cropped_count, detected_categories = detect_and_crop_objects(yolo_model, temp_image_path)

    # Fetch catalog images from local directory within detected categories
    catalog_images = fetch_images_from_local(categories=detected_categories)

    # Extract features from catalog images within detected categories
    catalog_features = extract_features_from_catalog_images(catalog_images, resnet_model, preprocess, detected_categories)

    # Extract features from cropped images
    feature_dict = extract_features_from_cropped_images("cropped_images", resnet_model, preprocess)

    # Generate recommendations
    recommendations = generate_recommendations(feature_dict, catalog_features)

    return JSONResponse(content=recommendations)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
