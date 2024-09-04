import os
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import io
import logging
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from azure.storage.blob import BlobServiceClient
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")

Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Define Product model
class Product(Base):
    __tablename__ = 'Products'
    ProductID = Column(Integer, primary_key=True, index=True)
    ProductName = Column(String, index=True)
    Description = Column(String)
    Price = Column(Float)
    ImageURL = Column(String)
    CreatedAt = Column(DateTime, default=datetime.utcnow)
    UpdatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Initialize Blob Service Client
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_name = "productimages"

# Function to add a product to the database
def add_product(db: Session, product_name: str, description: str, price: float, image_url: str):
    new_product = Product(
        ProductName=product_name,
        Description=description,
        Price=price,
        ImageURL=image_url,
    )
    db.add(new_product)
    db.commit()

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
def detect_and_crop_objects(model, image, cropped_images_dir='cropped_images', conf=0.2):
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

# Function to fetch all images and their descriptions from Azure Blob Storage
def fetch_images_and_descriptions_from_azure(db):
    images = {}
    container_client = blob_service_client.get_container_client(container_name)

    blobs = container_client.list_blobs()
    for blob in blobs:
        image_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob.name}"
        product = db.query(Product).filter(Product.ImageURL == image_url).first()
        if product:
            image = Image.open(io.BytesIO(container_client.download_blob(blob.name).readall())).convert('RGB')
            images[image_url] = {
                "image": image,
                "description": product.Description,
                "product_name": product.ProductName,
                "price": product.Price
            }

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

# Function to extract features from catalog images
def extract_features_from_catalog_images(catalog_images, model, preprocess):
    catalog_features = {}
    for image_url, data in catalog_images.items():
        features = extract_features(data["image"], model, preprocess)
        catalog_features[image_url] = features
        logger.info(f"Features extracted for {image_url}")
    logger.info("Features extracted for all catalog images.")
    return catalog_features

# Function to find similar items
def find_similar_items(features, catalog_features, db: Session, top_k=5):
    similarities = []
    for img_path, catalog_feature in catalog_features.items():
        similarity = cosine_similarity(features.reshape(1, -1), catalog_feature.reshape(1, -1))[0][0]
        similarities.append((img_path, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Fetch product details for the top_k similar items
    recommendations = []
    for img_path, _ in similarities[:top_k]:
        product = db.query(Product).filter(Product.ImageURL == img_path).first()
        if product:
            recommendations.append({
                "image_url": product.ImageURL,
                "description": product.Description,
                "product_name": product.ProductName,
                "price": product.Price
            })
    return recommendations

# Function to generate recommendations
def generate_recommendations(feature_dict, catalog_features, db):
    recommendations = {}
    for img_name, features in feature_dict.items():
        similar_items = find_similar_items(features, catalog_features, db)
        recommendations[img_name] = similar_items
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

    # Perform object detection and cropping
    cropped_count = detect_and_crop_objects(yolo_model, image)


    # Create a new database session
    db = SessionLocal()

    # Fetch all catalog images and descriptions from Azure Blob Storage
    catalog_images = fetch_images_and_descriptions_from_azure(db)


    # Extract features from catalog images
    catalog_features = extract_features_from_catalog_images(catalog_images, resnet_model, preprocess)

    # Extract features from cropped images
    feature_dict = extract_features_from_cropped_images("cropped_images", resnet_model, preprocess)


    # Generate recommendations
    recommendations = generate_recommendations(feature_dict, catalog_features, db)

    db.close()  # Close the database session

    return JSONResponse(content=recommendations)

if __name__ == "__main__":
    # Start the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)
