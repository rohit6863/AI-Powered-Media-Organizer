import os
import shutil
import zipfile
import torch
import joblib
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from flask import Flask, render_template, request, send_file

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = "uploads/raw_files"
ORGANIZED_FOLDER = "uploads/organized"
ZIP_PATH = "uploads/sorted_files.zip"

CLASSIFIER_PATH = "models/classifier_model.pkl"
CLASSES_PATH = "models/classes.npy"

# Load ML Models once when server starts to make requests fast
print("Loading Custom ML Model & CLIP Engine into Memory...")
try:
    clf = joblib.load(CLASSIFIER_PATH)
    classes = np.load(CLASSES_PATH)
    model_id = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    clip_model.eval()
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"⚠️ Error loading models. Did you run Phase 3? Error: {e}")


def cleanup_and_prepare_dirs():
    """Cleans up temporary directories for a fresh run."""
    if os.path.exists('uploads'):
        shutil.rmtree('uploads')
    
    os.makedirs(UPLOAD_FOLDER)
    os.makedirs(ORGANIZED_FOLDER)
    for c in classes:
        os.makedirs(os.path.join(ORGANIZED_FOLDER, str(c)), exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/guide')
def guide():
    return render_template('guide.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return "No files part", 400
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return "No selected file", 400

    cleanup_and_prepare_dirs()
    
    print(f"Received {len(files)} files. Starting Sorting Magic...")

    # Save and sort each file
    for file in files:
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            # Get safe filename and temporary path
            filename = os.path.basename(file.filename)
            temp_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(temp_path)
            
            try:
                # 1. Extract Features with CLIP
                image = Image.open(temp_path).convert("RGB")
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
                
                # 2. Predict Category
                prediction_idx = clf.predict(feature_vector)[0]
                predicted_class = classes[prediction_idx]
                
                # 3. Move to Organized Folder
                destination = os.path.join(ORGANIZED_FOLDER, str(predicted_class), filename)
                shutil.move(temp_path, destination)
                
            except Exception as e:
                print(f"Error sorting {filename}: {e}")

    # Zip the organized folder
    print("Zipping the sorted folders...")
    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(ORGANIZED_FOLDER):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, ORGANIZED_FOLDER)
                zipf.write(file_path, arcname)

    # Return the zip file to the user's browser for download
    return send_file(ZIP_PATH, as_attachment=True, download_name="organized_media.zip")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
