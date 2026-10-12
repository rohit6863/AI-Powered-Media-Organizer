import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

# 1. Load the pre-trained CLIP model from HuggingFace
print("Loading CLIP Model (This might take a minute the first time)...")
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

def classify_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return f"Error loading image: {e}"

    # 2. Define our custom, highly specific categories
    # This is the magic of CLIP - you just describe what you want in plain English!
    categories = [
        "A formal passport size portrait photo of a person with a plain background",
        "A beautiful, high-quality captured photograph or landscape",
        "A low-quality WhatsApp forwarded image with text and emojis, like a good morning quote",
        "A study note, textbook page, or document with lots of text",
        "An internet meme with funny text",
        "A random blurry or useless trash image"
    ]

    # 3. Process the image and the text prompts
    inputs = processor(text=categories, images=image, return_tensors="pt", padding=True)

    # 4. Get the model's predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 5. Calculate probabilities
    logits_per_image = outputs.logits_per_image 
    probs = logits_per_image.softmax(dim=1).numpy()[0]

    # Match probabilities to categories and find the highest one
    results = list(zip(categories, probs))
    results.sort(key=lambda x: x[1], reverse=True)
    
    best_category, highest_prob = results[0]
    
    # Map the long sentence back to a simple folder name
    if "passport" in best_category: folder = "Passport_Photos"
    elif "captured" in best_category: folder = "Captured_Photos"
    elif "WhatsApp" in best_category: folder = "WhatsApp_Trash"
    elif "study" in best_category: folder = "Docs_and_Notes"
    elif "meme" in best_category: folder = "Memes"
    else: folder = "Trash"

    return folder, highest_prob, results

if __name__ == "__main__":
    print("\n--- CLIP Image Classifier Ready ---")
    print("To test this, put an image inside the 'test_images' folder and update the path below.")
    
    # Example usage:
    # test_image_path = "../test_images/sample.jpg"
    # if os.path.exists(test_image_path):
    #     folder, prob, all_results = classify_image(test_image_path)
    #     print(f"\nPrediction: {folder} (Confidence: {prob:.2%})")
    # else:
    #     print(f"Please put an image at {test_image_path} to test!")
