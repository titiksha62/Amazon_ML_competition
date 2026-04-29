import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
import os
import sys

# --- 1. CONFIGURATION (Your Settings) ---
# Update IMAGE_DIR to the ABSOLUTE path if the relative path doesn't work!
TRAIN_DATA_PATH = 'train_image_id.csv' 
IMAGE_DIR = 'product_images'            
OUTPUT_FEATURES_PATH = 'embeds_image.csv'

# EfficientNetB0 specific settings
MODEL_NAME = 'EfficientNetB0'
IMAGE_SIZE = 224
BATCH_SIZE = 64
EMBEDDING_DIM = 1280

# --- 2. Build the Feature Extractor Model ---
print(f"TensorFlow Version: {tf.__version__}")
print(f"Configuring to use {MODEL_NAME} on CPU with batch size {BATCH_SIZE}...")

try:
    # Load EfficientNetB0, excluding the final classification layer (include_top=False)
    base_model = EfficientNetB0(
        weights='imagenet', 
        include_top=False, 
        pooling='avg', # This gives the 1280-dimension vector
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    feature_extractor = tf.keras.Model(
        inputs=base_model.input, 
        outputs=base_model.output
    )
    feature_extractor.trainable = False 
    print(f"Model loaded successfully. Output feature vector size: {EMBEDDING_DIM}")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)


# --- 3. Data Preparation and Image List (FIXED FOR FILENAME) ---
print("\nStep 3: Loading data and verifying image paths...")

try:
    df_config = pd.read_csv(TRAIN_DATA_PATH)
except FileNotFoundError:
    print(f"Error: Training data CSV not found at {TRAIN_DATA_PATH}")
    sys.exit(1)

# CRITICAL FIX: The image_id column (e.g., '51mo8htw') needs the file extension.
# We are assuming the images are JPEGs (.jpg) as a common standard. 
# ⚠️ CHANGE '.jpg' BELOW IF YOUR FILES ARE '.png' or another format!
df_config['image_filename'] = df_config['image_id'].astype(str) + '.jpg' 
print(f"Assuming image file extension is '.jpg'.")


# Create a list of (sample_id, full_file_path) tuples
image_list = []
# Resolve the IMAGE_DIR to an absolute path for better stability
resolved_image_dir = os.path.abspath(IMAGE_DIR)

for index, row in df_config.iterrows():
    # Construct the full path using the new, complete filename
    file_path = os.path.join(resolved_image_dir, row['image_filename']) 
    
    if os.path.exists(file_path):
        image_list.append((row['sample_id'], file_path))
    # else:
    #     # Uncomment this line if you still get 0 to debug the exact path
    #     # print(f"Missing file: {file_path}") 
    #     pass

print(f"Total entries found in CSV: {len(df_config)}")
print(f"Found {len(image_list)} image files available for processing.")

if len(image_list) == 0:
    print("\nFATAL ERROR: Found 0 images. Check your 'IMAGE_DIR' path and the file extension ('.jpg' vs '.png', etc.).")
    sys.exit(1)


# --- 4. Helper Function for Preprocessing ---
def load_and_preprocess_image(path):
    try:
        # Load image and resize to 224x224
        img = image.load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        # Convert to numpy array (float32)
        img_array = image.img_to_array(img, dtype='float32')
        # Expand dimensions for batch processing
        img_array = np.expand_dims(img_array, axis=0)
        # Apply standard EfficientNet pre-processing
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        # Fail gracefully for corrupted or unsupported images
        return None

# --- 5. Batch Feature Extraction Loop ---
print("\nStep 4: Starting feature extraction (this will take time on CPU)...")

all_embeddings = []
all_sample_ids = []

# Use tqdm for a progress bar
for i in tqdm(range(0, len(image_list), BATCH_SIZE)):
    batch_list = image_list[i:i + BATCH_SIZE]
    
    batch_images = []
    batch_ids = []
    
    # Preprocess the images in the current batch
    for sample_id, path in batch_list:
        processed_img = load_and_preprocess_image(path)
        if processed_img is not None:
            batch_images.append(processed_img)
            batch_ids.append(sample_id)
            
    if not batch_images:
        continue

    # Stack the list of arrays into a single batch tensor
    batch_input = np.vstack(batch_images)
    
    # Predict features using the model
    # Note: On CPU, set batch_size=None or leave it out if performance is poor
    batch_features = feature_extractor.predict(batch_input, verbose=0)
    
    # Store results
    all_embeddings.append(batch_features)
    all_sample_ids.extend(batch_ids)

# --- 6. Finalization and Save ---

# Combine all batch results into a final NumPy array
final_features_array = np.vstack(all_embeddings)
print(f"\nSuccessfully extracted features for {final_features_array.shape[0]} samples.")

# Create the output DataFrame
feature_cols = [f'img_feat_{j}' for j in range(EMBEDDING_DIM)]
df_features = pd.DataFrame(final_features_array, columns=feature_cols)
df_features['sample_id'] = all_sample_ids

# Reorder and save
df_features = df_features[['sample_id'] + feature_cols]
df_features.to_csv(OUTPUT_FEATURES_PATH, index=False)
print(f"\n--- SUCCESS ---")
print(f"Image features saved to {OUTPUT_FEATURES_PATH}")
print(f"File contains {df_features.shape[0]} rows and {df_features.shape[1]} columns (1 for ID + 1280 features).")