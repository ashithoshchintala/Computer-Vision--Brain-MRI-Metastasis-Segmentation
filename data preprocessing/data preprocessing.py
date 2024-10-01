import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# CLAHE Preprocessing
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

# Load images and masks
def load_images_and_masks(image_dir, mask_dir):
    images, masks = [], []
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)

        if os.path.exists(img_path) and os.path.exists(mask_path):
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Apply CLAHE
            image = apply_clahe(image)

            images.append(image)
            masks.append(mask)

    return np.array(images), np.array(masks)

# Split the dataset
def prepare_data(image_dir, mask_dir):
    images, masks = load_images_and_masks(image_dir, mask_dir)
    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)
    return (X_train, y_train), (X_test, y_test)

# Normalize images
def normalize_images(images):
    images = images.astype('float32') / 255.0
    return images[..., np.newaxis]

# Example use
   # Update this with the path of your masks
image_dir='E:\important certificates\GITHUB\Computer-Vision--Brain-MRI-Metastasis-Segmentation\data preprocessing\Data'
mask_dir='E:\important certificates\GITHUB\Computer-Vision--Brain-MRI-Metastasis-Segmentation\data preprocessing\Data'
(X_train, y_train), (X_test, y_test) = prepare_data(image_dir, mask_dir)
X_train = normalize_images(X_train)
X_test = normalize_images(X_test)
