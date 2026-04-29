import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
INPUT_ENGINEERED_PATH = 'data_intermediates/test_intermediate_engineered_features.csv'
TEXT_ASSETS_PATH = 'assets_trained/trained_text_assets.pkl' # TFIDF and SVD from training
TEST_IMAGE_FOLDER = 'test_product_images' 
OUTPUT_MASTER_PATH = 'data_intermediates/master_test_features.csv'

# Constants (MUST BE IDENTICAL TO TRAINING)
IMAGE_SIZE = 224
BATCH_SIZE = 64
EMBEDDING_DIM = 1280  # EfficientNetB0 output dimension
TEXT_EMBEDDING_DIM = 500 # SVD output dimension

# --- 2. Load Data, Models, and Trained Assets ---
print("Step 2: Loading engineered test data and trained text assets...")
try:
    df_test = pd.read_csv(INPUT_ENGINEERED_PATH)
    # Load the crucial text transformers
    with open(TEXT_ASSETS_PATH, 'rb') as f:
        assets = pickle.load(f)
        tfidf_vectorizer = assets['tfidf_vectorizer']
        svd_model = assets['svd_model']
except FileNotFoundError as e:
    print(f"Error: Required file not found: {e}")
    print("Ensure 05a was run and trained assets were saved by 02.")
    sys.exit(1)

# Prepare text data for transformation
text_data_test = df_test['cleaned_content'].fillna('')


# --- 3. Text Embedding (Using .transform()) ---
print("Step 3: Applying trained TF-IDF and SVD models...")

# CRITICAL: Use .transform(), not .fit_transform()
tfidf_matrix_test = tfidf_vectorizer.transform(tqdm(text_data_test, desc="TF-IDF Transform"))
text_embeddings_test = svd_model.transform(tfidf_matrix_test)

text_feature_cols = [f'text_feat_{i}' for i in range(TEXT_EMBEDDING_DIM)]
df_text_features = pd.DataFrame(text_embeddings_test, columns=text_feature_cols)
# Add sample_id for later merge
df_text_features['sample_id'] = df_test['sample_id'].values


# --- 4. Image Embedding Preparation and Setup ---

# Image Model Setup (Same as 03_embed_images.py)
base_model = EfficientNetB0(
    weights='imagenet', 
    include_top=False, 
    pooling='avg', 
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)
base_model.trainable = False
feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)


def load_and_preprocess_image(path):
    """Loads, resizes, and preprocesses a single image file."""
    try:
        img = image.load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.expand_dims(image.img_to_array(img, dtype='float32'), axis=0)
        return preprocess_input(img_array)
    except Exception: return None

# Prepare list of (sample_id, full_file_path) tuples
image_list = []
resolved_image_dir = os.path.abspath(TEST_IMAGE_FOLDER)

for index, row in df_test.iterrows():
    # Use 'image_id' (filename without extension) to construct the path
    image_found = False
    for ext in ['.jpg', '.jpeg', '.png']:
        file_name = row['image_id'] + ext
        file_path = os.path.join(resolved_image_dir, file_name)
        if os.path.exists(file_path):
            image_list.append((row['sample_id'], file_path))
            image_found = True
            break
    
    if not image_found:
        pass # Skip if image not found

print(f"Total available test images found for embedding: {len(image_list)}")


# --- 5. Image Embedding Execution (Batch Processing) ---
print("\nStep 5: Generating image embeddings...")

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
            
    if not batch_images: continue
    
    batch_input = np.vstack(batch_images)
    batch_features = feature_extractor.predict(batch_input, verbose=0)
    
    all_embeddings.append(batch_features)
    all_sample_ids.extend(batch_ids)

if all_embeddings:
    final_features_array = np.vstack(all_embeddings)
else:
    # Handle case where no images are found (shouldn't happen with verified IDs)
    final_features_array = np.empty((0, EMBEDDING_DIM))

image_feature_cols = [f'img_feat_{j}' for j in range(EMBEDDING_DIM)]
df_image_features = pd.DataFrame(final_features_array, columns=image_feature_cols)
df_image_features['sample_id'] = all_sample_ids


# --- 6. Final Merge and Preparation of X_test ---
print("\nStep 6: Merging all test features...")

# Start with the manual features (dropping 'cleaned_content' and 'image_link')
df_master_test = df_test.drop(columns=['cleaned_content', 'image_link'], errors='ignore').copy()

# 6a. Merge Text Embeds (should align by index as text_features was built from df_test)
# Using merge for safety, though concat is also possible if index is ensured
df_master_test = df_master_test.merge(df_text_features, on='sample_id', how='left')

# 6b. Merge Image Embeds (Left Join to keep all test samples)
df_master_test = df_master_test.merge(df_image_features, on='sample_id', how='left')

# 6c. Handle missing embeddings (Fill NaNs with 0, as done in training)
embedding_cols = [col for col in df_master_test.columns if col.startswith(('text_feat_', 'img_feat_'))]
df_master_test[embedding_cols] = df_master_test[embedding_cols].fillna(0)

# The final master test data is ready for prediction
os.makedirs(os.path.dirname(OUTPUT_MASTER_PATH), exist_ok=True)
df_master_test.to_csv(OUTPUT_MASTER_PATH, index=False)

print(f"\n--- SUCCESS: Step 05c Complete ---")
print(f"Final master test feature file saved to {OUTPUT_MASTER_PATH}")