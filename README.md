# 📂 AI-Powered Media Organizer

An intelligent web application that uses Deep Learning and Computer Vision to automatically organize messy folders of mixed images into neat, semantic categories.

Built with **OpenAI CLIP**, **Scikit-Learn**, and **Flask**, featuring a clean custom Neumorphic UI.

---

## 🚀 Features

- **Semantic Understanding**: Uses OpenAI's CLIP (Vision Transformer) embeddings rather than surface-level file metadata.
- **7 Target Categories**:
  - `docs_and_notes` — Physical paperwork, study notes, forms, certificates
  - `id_cards` — Identity cards and official IDs
  - `memes` — Humorous images and internet memes
  - `miscellaneous` — Textures, wallpapers, abstract media
  - `photos` — Real-world photos, portraits, scenery
  - `screenshots` — Device captures, mobile/desktop screens
  - `trash` — Blurry, accidental, or low-quality camera captures
- **One-Click Folder Sorting**: Select a folder, and the AI categorizes all media and serves back an organized `.zip` archive.
- **Interactive Web Interface**: Custom brutalist/neumorphic UI with 3D button interactions and a built-in project guide page (`/guide`).

---

## 🧠 Architecture & Methodology

```
User Uploads Folder
       │
       ▼
Flask Backend Server
       │
       ▼
OpenAI CLIP (vit-base-patch32)
   [Feature Extraction]
       │  (768-dimensional dense vector)
       ▼
Logistic Regression Classifier
   [Trained on custom feature embeddings]
       │
       ▼
Sorting & Zipping Engine ──► User Downloads sorted_files.zip
```

1. **Feature Extraction**: Images are passed through the frozen `openai/clip-vit-base-patch32` model to extract a 768-dimensional semantic embedding.
2. **Classification**: A tuned Logistic Regression model maps embeddings to the 7 target classes.
3. **Hard Negative Mining**: Handled class ambiguity (e.g., distinguishing white web screenshots from text documents) by feeding boundary edge cases into training, achieving **90.00% validation accuracy**.

---

## 📊 Performance Evaluation

| Category | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **docs_and_notes** | 0.80 | 0.80 | 0.80 |
| **id_cards** | 0.80 | 1.00 | 0.89 |
| **memes** | 1.00 | 1.00 | 1.00 |
| **miscellaneous** | 1.00 | 1.00 | 1.00 |
| **photos** | 1.00 | 1.00 | 1.00 |
| **screenshots** | 1.00 | 0.67 | 0.80 |
| **trash** | 1.00 | 1.00 | 1.00 |
| **Weighted Average** | **0.91** | **0.90** | **0.90** |

Overall Validation Accuracy: **90.00%**

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/aiml-media-organizer.git
cd aiml-media-organizer
```

### 2. Create a virtual environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5000`.

---

## 📁 Project Structure

```
Aiml_project/
├── app.py                  # Main Flask application & inference routes
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker container configuration
├── dataset/                # Training dataset by category
│   └── train/
├── models/
│   ├── classifier_model.pkl# Trained Logistic Regression model
│   ├── classes.npy         # Target class names
│   └── confusion_matrix.png# Evaluation confusion matrix heatmap
├── src/
│   ├── extract_features.py # CLIP feature extraction script
│   ├── train_classifier.py # Model training & evaluation script
│   └── sort_my_folder.py   # CLI version of folder sorter
├── static/                 # CSS styling & illustrations
│   ├── style.css
│   └── images/
└── templates/              # HTML frontend templates
    ├── index.html
    └── guide.html
```

---

## 📜 License
This project is open-source under the MIT License.
