# %% [markdown]
# # AIML Project: Media Classifier Training
# This file can be run as a standard Python script, or you can run the cells 
# interactively in VS Code just like a Jupyter Notebook!

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# %%
print("--- Phase 3: Model Training ---")
# 1. Load the extracted features
features_path = "../models/X_features.npy"
labels_path = "../models/y_labels.npy"
classes_path = "../models/classes.npy"

if not os.path.exists(features_path):
    raise FileNotFoundError("Could not find extracted features. Did Phase 2 finish?")

X = np.load(features_path)
y = np.load(labels_path)
classes = np.load(classes_path)

print(f"Loaded {X.shape[0]} images across {len(classes)} classes: {list(classes)}")

# %%
# 2. Split into Training and Testing sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training on {X_train.shape[0]} images, testing on {X_test.shape[0]} images.")

# %%
# 3. Train the Classifier (Linear Probing approach)
print("\nTraining the model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Training complete!")

# %%
# 4. Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=classes))

# %%
# 5. Plot Confusion Matrix (Great for the project report!)
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix - Media Classifier')
plt.ylabel('Actual Category')
plt.xlabel('Predicted Category')
plt.tight_layout()

# Save the plot
plot_path = "../models/confusion_matrix.png"
plt.savefig(plot_path)
print(f"\nSaved confusion matrix graph to {plot_path}")

# Display the plot if running interactively
# plt.show()

# %%
# 6. Save the trained model for Phase 4 (The final application)
model_path = "../models/classifier_model.pkl"
joblib.dump(model, model_path)
print(f"Saved trained model to {model_path}")
print("Ready for Phase 4!")
