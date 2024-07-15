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
def detect_and_crop_objects(model, image_url, cropped_images_dir='cropped_images', conf=0.8):
    clear_directory(cropped_images_dir)
    results = model(source=image_url, conf=conf)
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
            print(f"Cropped image saved to {cropped_img_path}")
    return results, cropped_count, detected_categories

# Function to extract features from cropped images
def extract_features_from_cropped_images(cropped_images_dir, model, preprocess):
    feature_dict = {}
    for img_path in os.listdir(cropped_images_dir):
        full_img_path = os.path.join(cropped_images_dir, img_path)
        if os.path.isfile(full_img_path):  # Ensure it's a file
            features = extract_features(full_img_path, model, preprocess)
            feature_dict[img_path] = features
    print("Features extracted for all cropped images.")
    return feature_dict

# Function to extract features from catalog images within specific categories
def extract_features_from_catalog_images(catalog_dir, model, preprocess, categories):
    catalog_features = {}
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

    for category in categories:
        category_path = os.path.join(catalog_dir, category)
        if os.path.isdir(category_path):  # Ensure it's a directory
            for img_path in os.listdir(category_path):
                full_img_path = os.path.join(category_path, img_path)
                if os.path.isfile(full_img_path) and os.path.splitext(full_img_path)[1].lower() in image_extensions:  # Ensure it's an image file
                    features = extract_features(full_img_path, model, preprocess)
                    catalog_features[f"{category}/{img_path}"] = features
    print("Features extracted for all catalog images in detected categories.")
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
        recommendations[img_name] = similar_items
    print("Recommendations generated.")
    return recommendations





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
results, cropped_count, detected_categories = detect_and_crop_objects(yolo_model, "/Users/manya./repos/HouseRizz/617924fd17bb27257e45e0cc47f72133.jpg")
print(f"Detected and cropped {cropped_count} objects in categories: {detected_categories}")

# Extract features from cropped images
feature_dict = extract_features_from_cropped_images('cropped_images', resnet_model, preprocess)
print(f"Extracted features from cropped images: {len(feature_dict)}")

# Extract features from catalog images in detected categories only
catalog_features = extract_features_from_catalog_images('/Users/manya./PycharmProjects/Model/BaseSimilarityModel/images', resnet_model, preprocess, detected_categories)
print(f"Extracted features from catalog images in categories: {detected_categories}")

# Generate recommendations
recommendations = generate_recommendations(feature_dict, catalog_features)
print(f"Generated recommendations for {len(recommendations)} detected objects")

# Display recommendations
display_recommendations(recommendations, 'cropped_images', '/Users/manya./PycharmProjects/Model/BaseSimilarityModel/images')

