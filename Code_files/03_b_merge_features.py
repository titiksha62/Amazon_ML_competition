import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
INPUT_MANUAL_TEXT_PATH = 'data_intermediates/intermediate_engineered_features.csv'
INPUT_TEXT_EMBEDS_PATH = 'data_intermediates/embeds_text_engineered.csv'
INPUT_IMAGE_EMBEDS_PATH = 'data_intermediates/embeds_image.csv'
# The final file for training
FINAL_TRAIN_OUTPUT_PATH = 'data_intermediates/master_train_data.csv'

# --- 2. Load All Feature Files ---
print("Step 2: Loading all training feature components...")

try:
    # 2a. Manual and Cleaned Text Features (Contains log_price and manual features)
    df_manual_text = pd.read_csv(INPUT_MANUAL_TEXT_PATH)
    
    # 2b. Text Embeddings (Contains text_feat_...)
    df_text_embeds = pd.read_csv(INPUT_TEXT_EMBEDS_PATH)
    
    # 2c. Image Embeddings (Contains img_feat_...)
    df_image_embeds = pd.read_csv(INPUT_IMAGE_EMBEDS_PATH)
    
except FileNotFoundError as e:
    print(f"Error: Required input file not found: {e}")
    print("Ensure all previous scripts (01, 02, 03_embed_images) were run successfully.")
    sys.exit(1)

# --- 3. Merging Features ---
print("Step 3: Merging all feature sets...")

# Start with the full dataset containing manual features and target
df_master = df_manual_text.copy()

# Drop the intermediate 'cleaned_content' and 'image_link' before final merge
df_master = df_master.drop(columns=['cleaned_content', 'image_link'], errors='ignore') 

# Merge Text Embeddings (on 'sample_id')
df_master = df_master.merge(df_text_embeds, on='sample_id', how='left')

# Merge Image Embeddings (on 'sample_id'). Left merge preserves all rows.
df_master = df_master.merge(df_image_embeds, on='sample_id', how='left')

# --- 4. Final Cleaning and Saving ---
print("Step 4: Imputing missing image features...")

# Identify embedding columns (image features will have NaNs where images were missing)
embedding_cols = [col for col in df_master.columns if col.startswith(('text_feat_', 'img_feat_'))]

# Impute Missing Embeddings: Fill all NaN values in the embedding columns with 0.
df_master[embedding_cols] = df_master[embedding_cols].fillna(0)

# Drop the redundant 'price' column (keep 'log_price' as target)
df_master = df_master.drop(columns=['price']) 

# Save the final master training file
os.makedirs(os.path.dirname(FINAL_TRAIN_OUTPUT_PATH), exist_ok=True)
df_master.to_csv(FINAL_TRAIN_OUTPUT_PATH, index=False)

print(f"\n--- SUCCESS: Step 03a Complete (Merge Features) ---")
print(f"Final training data saved to {FINAL_TRAIN_OUTPUT_PATH}")