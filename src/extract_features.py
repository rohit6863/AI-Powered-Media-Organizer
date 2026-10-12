import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# Paths
DATASET_DIR = "../dataset/train"
OUTPUT_FEATURES = "../models/X_features.npy"
OUTPUT_LABELS = "../models/y_labels.npy"
OUTPUT_CLASSES = "../models/classes.npy"

def extract_features():
    print("Loading CLIP Model...")
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    
    # Put model in evaluation mode
    model.eval()

    X = []
    y = []
    
    # Get all folder names (these are our class labels)
    classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    print(f"Found classes: {classes}")

    if not classes:
        print("Error: No class folders found in dataset/train/")
        return

    # Loop through each class folder
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(DATASET_DIR, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        print(f"Processing {len(image_files)} images in '{class_name}'...")
        
        for img_file in tqdm(image_files):
            img_path = os.path.join(class_path, img_file)
            
            try:
                # Open image and convert to RGB
                image = Image.open(img_path).convert("RGB")
                
                # Preprocess the image for CLIP
                inputs = processor(images=image, return_tensors="pt")
                
                # Extract the 512-dim visual embedding
                with torch.no_grad():
                    outputs = model.get_image_features(**inputs)
                    
                    # Depending on the transformers version, it might return a tensor or an object
                    if hasattr(outputs, "image_embeds"):
                        image_features = outputs.image_embeds
                    elif hasattr(outputs, "last_hidden_state"):
                        image_features = outputs.last_hidden_state[:, 0, :]
                    else:
                        image_features = outputs # If it's already a tensor
                
                # Normalize the features (standard practice for CLIP embeddings)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
                X.append(image_features.numpy().flatten())
                y.append(class_idx)
                
            except Exception as e:
                print(f"\nSkipping {img_file} due to error: {e}")

    # Convert lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    classes = np.array(classes)
    
    if len(X) == 0:
        print("\nNo valid images found to process. Please add images to the folders.")
        return

    # Save to disk
    np.save(OUTPUT_FEATURES, X)
    np.save(OUTPUT_LABELS, y)
    np.save(OUTPUT_CLASSES, classes)
    
    print(f"\nSuccessfully extracted features for {len(X)} images!")
    print(f"Features shape: {X.shape} (Number of images, 512 dimensions)")
    print(f"Saved matrices to {os.path.abspath('../models/')}")

if __name__ == "__main__":
    extract_features()
