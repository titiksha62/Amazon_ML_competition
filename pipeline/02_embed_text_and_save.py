import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
from tqdm import tqdm
import sys
import os
import warnings
warnings.filterwarnings("ignore") 

# --- 1. CONFIGURATION ---
INPUT_ENGINEERED_PATH = 'data_intermediates/intermediate_engineered_features.csv'
TEXT_COLUMN_CLEAN = 'cleaned_content'
OUTPUT_FEATURES_PATH = 'data_intermediates/embeds_text_engineered_2.csv' 
ASSETS_PATH = 'assets_trained/trained_text_assets.pkl'

# TF-IDF & SVD Parameters (Ensure these match your desired settings)
MAX_FEATURES = 100000
SVD_COMPONENTS = 500
MIN_DF = 5 

# --- 2. Load Engineered Data ---
print(f"Step 2: Loading cleaned data from {INPUT_ENGINEERED_PATH}...")
try:
    df_engineered = pd.read_csv(INPUT_ENGINEERED_PATH)
except FileNotFoundError:
    print(f"Error: Intermediate file not found at {INPUT_ENGINEERED_PATH}. Run 01_engineer_text.py first.")
    sys.exit(1)

# Ensure 'cleaned_content' is handled as string and fill NaNs (safety)
text_data = df_engineered[TEXT_COLUMN_CLEAN].fillna('')


# --- 3. TF-IDF Vectorization (Fit & Transform) ---
print(f"\nStep 3: Creating TF-IDF vectors...")

tfidf_vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=(1, 2), 
    stop_words='english',
    min_df=MIN_DF
)

# CRITICAL: Use .fit_transform() to learn the vocabulary and weights
tfidf_matrix = tfidf_vectorizer.fit_transform(tqdm(text_data, desc="TF-IDF Fit/Transform"))


# --- 4. Dimensionality Reduction with Truncated SVD (Fit & Transform) ---
print(f"\nStep 4: Reducing dimensions using Truncated SVD (Components: {SVD_COMPONENTS})...")

svd_model = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=42)

# CRITICAL: Use .fit_transform() to learn the SVD components
text_embeddings = svd_model.fit_transform(tfidf_matrix)


# --- 5. Finalize and Save the Engineered/Text Features ---
print("\nStep 5: Saving text features...")

# Create DataFrame for SVD Embeddings
feature_cols = [f'text_feat_{i}' for i in range(SVD_COMPONENTS)]
df_text_features = pd.DataFrame(text_embeddings, columns=feature_cols)

# Ensure alignment and merge
# Note: df_engineered already contains the manual features
df_text_output = pd.concat([df_engineered[['sample_id']], df_text_features], axis=1)

# Save the text features
os.makedirs(os.path.dirname(OUTPUT_FEATURES_PATH), exist_ok=True)
df_text_output.to_csv(OUTPUT_FEATURES_PATH, index=False)


# --- 6. CRITICAL: SAVE TRAINED ASSETS ---
print("\nStep 6: Saving trained text models (TFIDF and SVD)...")
trained_assets = {
    'tfidf_vectorizer': tfidf_vectorizer,
    'svd_model': svd_model
}
os.makedirs(os.path.dirname(ASSETS_PATH), exist_ok=True)
with open(ASSETS_PATH, 'wb') as f:
    pickle.dump(trained_assets, f)

print(f"\n--- SUCCESS: Step 2 Complete ---")
print(f"Text features saved to {OUTPUT_FEATURES_PATH}")
print(f"Trained models saved to {ASSETS_PATH} for test reuse.")