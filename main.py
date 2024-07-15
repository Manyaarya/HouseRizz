import os
import cv2 # type: ignore
import torch # type: ignore
import torchvision.models as models # type: ignore
import torchvision.transforms as transforms # type: ignore
from PIL import Image # type: ignore
from ultralytics import YOLO # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore

