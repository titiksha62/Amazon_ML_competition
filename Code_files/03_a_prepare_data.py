import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Set up TensorFlow to prevent memory issues, if necessary
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if physical_devices:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)

# --- 1. CONFIGURATION ---
INPUT_ENGINEERED_PATH = 'data_intermediates/intermediate_engineered_features.csv'
IMAGE_FOLDER = 'product_images' # Path to your training images
OUTPUT_FEATURES_PATH = 'data_intermediates/embeds_image.csv'

# Constants for Image Model (MUST be consistent across train/test)
IMAGE_SIZE = 224
BATCH_SIZE = 64
EMBEDDING_DIM = 1280 # Output feature size of EfficientNetB0 (pooling='avg')

# --- 2. Utility Function: Load and Preprocess Image ---

def load_and_preprocess_image(path):
    """Loads, resizes, and preprocesses a single image file."""
    try:
        # Load image with target size
        img = image.load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        # Convert to array, add batch dimension
        img_array = np.expand_dims(image.img_to_array(img, dtype='float32'), axis=0)
        # Apply EfficientNet preprocessing
        return preprocess_input(img_array)
    except Exception:
        # Returns None if file is corrupt or path is wrong
        return None

# --- 3. Load Data and Prepare Image List ---
print(f"Step 3: Loading engineered data from {INPUT_ENGINEERED_PATH}...")
try:
    df_engineered = pd.read_csv(INPUT_ENGINEERED_PATH)
except FileNotFoundError:
    print(f"Error: Intermediate file not found at {INPUT_ENGINEERED_PATH}. Run 01/02 first.")
    sys.exit(1)

# Prepare list of (sample_id, full_file_path) tuples
image_list = []
resolved_image_dir = os.path.abspath(IMAGE_FOLDER)

# Use 'image_id' (filename without extension) to construct the path
for index, row in df_engineered.iterrows():
    # Attempt to determine image extension based on common types, assuming one exists
    # If your image_id includes the extension, simplify this: os.path.join(resolved_image_dir, row['image_id'])
    
    # We will iterate through common extensions until we find a match
    image_found = False
    for ext in ['.jpg', '.jpeg', '.png']:
        file_name = row['image_id'] + ext
        file_path = os.path.join(resolved_image_dir, file_name)
        if os.path.exists(file_path):
            image_list.append((row['sample_id'], file_path))
            image_found = True
            break
    
    if not image_found:
        # If the file wasn't found (e.g., download failed, different extension), skip it.
        # This will create NaNs when features are merged, which will be filled with 0 later.
        pass 

print(f"Total available images found for embedding: {len(image_list)}")


# --- 4. Image Embedding Model Setup ---
print("\nStep 4: Setting up EfficientNetB0 feature extractor...")

# Load pre-trained EfficientNetB0, excluding the classification head (include_top=False)
base_model = EfficientNetB0(
    weights='imagenet', 
    include_top=False, 
    pooling='avg', # Global Average Pooling to reduce output to 1D vector (1280 features)
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)
# Freeze the layers; we only need the pre-trained features
base_model.trainable = False
feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)


# --- 5. Image Embedding Execution (Batch Processing) ---
print("\nStep 5: Generating embeddings...")

all_embeddings = []
all_sample_ids = []

for i in tqdm(range(0, len(image_list), BATCH_SIZE), desc="Generating Embeddings"):
    batch_list = image_list[i:i + BATCH_SIZE]
    batch_images = []
    batch_ids = []

    for sample_id, path in batch_list:
        processed_img = load_and_preprocess_image(path)
        if processed_img is not None:
            batch_images.append(processed_img)
            batch_ids.append(sample_id)
            
    if not batch_images:
        continue # Skip if batch is empty (e.g., all files in batch were corrupt)

    # Stack images and predict features
    batch_input = np.vstack(batch_images)
    batch_features = feature_extractor.predict(batch_input, verbose=0)
    
    all_embeddings.append(batch_features)
    all_sample_ids.extend(batch_ids)

# --- 6. Finalize and Save Image Features ---
print("\nStep 6: Saving image features...")

final_features_array = np.vstack(all_embeddings)
feature_cols = [f'img_feat_{j}' for j in range(EMBEDDING_DIM)]

df_image_features = pd.DataFrame(final_features_array, columns=feature_cols)
df_image_features['sample_id'] = all_sample_ids

os.makedirs(os.path.dirname(OUTPUT_FEATURES_PATH), exist_ok=True)
df_image_features.to_csv(OUTPUT_FEATURES_PATH, index=False)

print(f"\n--- SUCCESS: Step 3 Complete ---")
print(f"Image features saved to {OUTPUT_FEATURES_PATH}")
print("Next Step: Merge all features in 03_prepare_data.py!")