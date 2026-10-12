import os
import shutil
import torch
import joblib
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# --- Configuration ---
TARGET_FOLDER = "../test_images"  # The messy folder to sort
OUTPUT_FOLDER = "../test_images/organized" # Where to put the sorted files

# Model paths
CLASSIFIER_PATH = "../models/classifier_model.pkl"
CLASSES_PATH = "../models/classes.npy"

def sort_folder():
    print("\n" + "="*40)
    print("--- Phase 4: AI Folder Sorter ---")
    print("="*40)
    
    # 1. Load the trained Machine Learning model
    if not os.path.exists(CLASSIFIER_PATH):
        print("Error: Trained model not found. Run Phase 3 first.")
        return
        
    print("Loading your custom trained AI model...")
    clf = joblib.load(CLASSIFIER_PATH)
    classes = np.load(CLASSES_PATH)
    
    # 2. Load the CLIP Feature Extractor
    print("Loading CLIP Vision engine...")
    model_id = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    clip_model.eval()

    # 3. Prepare directories
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    for c in classes:
        os.makedirs(os.path.join(OUTPUT_FOLDER, str(c)), exist_ok=True)

    # Find all images in the target folder (ignoring subdirectories)
    images_to_sort = [f for f in os.listdir(TARGET_FOLDER) 
                      if os.path.isfile(os.path.join(TARGET_FOLDER, f)) and 
                         f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                         
    if len(images_to_sort) == 0:
        print(f"\nNo images found in '{TARGET_FOLDER}'. Please add some messy images to test!")
        return

    print(f"\nFound {len(images_to_sort)} images to sort. Let's go!\n")

    # 4. Process and sort each image
    for img_file in tqdm(images_to_sort):
        img_path = os.path.join(TARGET_FOLDER, img_file)
        
        try:
            # Extract features (Just like Phase 2)
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = clip_model.get_image_features(**inputs)
                
                if hasattr(outputs, "image_embeds"):
                    features = outputs.image_embeds
                elif hasattr(outputs, "last_hidden_state"):
                    features = outputs.last_hidden_state[:, 0, :]
                else:
                    features = outputs
            
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            feature_vector = features.numpy()
            
            # Predict the category using YOUR trained model (Just like Phase 3)
            prediction_idx = clf.predict(feature_vector)[0]
            predicted_class = classes[prediction_idx]
            
            # Copy the file to the correct folder (using copy2 to protect your original files during testing)
            destination = os.path.join(OUTPUT_FOLDER, str(predicted_class), img_file)
            shutil.copy2(img_path, destination) 
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

    print("\n[SUCCESS] Sorting Complete!")
    print(f"Check your beautifully organized files in: {os.path.abspath(OUTPUT_FOLDER)}")

if __name__ == "__main__":
    sort_folder()
