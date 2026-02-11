import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURATION ---
MODEL_PATH = 'models/ultimate_holistic_model.keras'  # Check if your model is in 'models/' or root
TEST_DIR = 'fer2013/test'                            # Path to your test dataset
IMG_SIZE = (48, 48)
BATCH_SIZE = 64

# --- CHECK PATHS ---
if not os.path.exists(MODEL_PATH):
    # Try root folder if not in models/
    if os.path.exists('ultimate_holistic_model.keras'):
        MODEL_PATH = 'ultimate_holistic_model.keras'
    else:
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        exit()

if not os.path.exists(TEST_DIR):
    print(f"❌ Error: Test dataset not found at '{TEST_DIR}'.")
    print("   Please download the FER2013 dataset and place the 'test' folder inside 'fer2013/'.")
    exit()

print(f"--- 🔄 Loading Model from: {MODEL_PATH} ---")
try:
    model = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit()

# --- PREPARE DATA ---
print(f"--- 📂 Loading Test Images from: {TEST_DIR} ---")
test_datagen = ImageDataGenerator(rescale=1./255)

try:
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False 
    )
except Exception as e:
    print(f"❌ Error loading images: {e}")
    exit()

# --- RUN EVALUATION ---
print("\n--- ⚡ Starting Accuracy Test (This may take 1-2 minutes) ---")
loss, accuracy = model.evaluate(test_generator, verbose=1)

print("\n" + "="*30)
print(f"🏆 FINAL ACCURACY: {accuracy * 100:.2f}%")
print("="*30 + "\n")

# --- DETAILED REPORT ---
print("--- 📊 Generating Detailed Classification Report ---")
predictions = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print(classification_report(true_classes, predicted_classes, target_names=class_labels))