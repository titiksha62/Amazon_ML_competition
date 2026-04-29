import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import sys
import warnings
warnings.filterwarnings("ignore") # Suppress FutureWarnings from scikit-learn

# --- 1. CONFIGURATION ---
TRAIN_DATA_PATH = 'train_image_id.csv'   # Your main data file
TEXT_COLUMN = 'catalog_content'
OUTPUT_FEATURES_PATH = 'embeds_text_engineered.csv'

# TF-IDF & SVD Parameters
MAX_FEATURES = 100000        
SVD_COMPONENTS = 500         
MIN_DF = 5                   

# Brand/Unit Extraction Lists
# NOTE: These lists are simplified. For a competition, you'd expand these heavily.
COMMON_UNITS = r'(oz|fl oz|lb|kg|g|mg|ml|l|count|pack|ct|pair|set|unit|sheets|bags)'
BRAND_KEYWORDS = [
    'Brand Name', 'Manufacturer', 'By:', 'Sold by', 'Visit the', 'by'
]

# --- 2. Load Data and Initial Preprocessing ---
print(f"Step 2: Loading data from {TRAIN_DATA_PATH}...")

try:
    # Load all relevant columns
    df = pd.read_csv(TRAIN_DATA_PATH)
except FileNotFoundError:
    print(f"Error: Training data CSV not found at {TRAIN_DATA_PATH}")
    sys.exit(1)

df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna('').astype(str)
print(f"Total records loaded: {len(df)}")


# --- 3. FEATURE ENGINEERING (Extraction of Brand, Name, Unit) ---
print("\nStep 3: Performing custom feature engineering (Brand, Name, Unit)...")

# --- 3a. Text Cleaning Function for Embeddings ---
def clean_for_embeddings(text):
    """Clean text for TF-IDF/SVD: Remove noise but keep important words."""
    text = text.lower()
    # Remove HTML tags (if any)
    text = re.sub(r'<.*?>', '', text)
    # Remove common product prefixes/noise words
    text = re.sub(r'(item name|product name|bullet point|product details|description)', ' ', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Replace multiple spaces with a single space and strip
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply cleaning for the embeddings column
df['cleaned_content'] = [clean_for_embeddings(text) for text in tqdm(df[TEXT_COLUMN], desc="Text Cleaning")]

# --- 3b. Extract Product Name (Usually the first line/sentence) ---
def extract_product_name(text):
    match = re.search(r'Item Name: (.*)', text)
    if match:
        name = match.group(1).split('.')[0].strip()
        # Clean up common debris
        name = re.sub(r'\(.*\)', '', name).strip()
        return name
    return None

df['product_name'] = df[TEXT_COLUMN].apply(extract_product_name).fillna('')

# --- 3c. Extract Unit and Value (Size/Quantity) ---
# Finds patterns like '12 oz' or '5 pack'
def extract_unit_and_value(text):
    # Regex to find a number followed by a unit, possibly with spaces/hyphens
    match = re.search(r'(\d+\.?\d*)\s*[-]*\s*(' + COMMON_UNITS + r')', text.lower())
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        return value, unit
    return None, None

df[['unit_value', 'unit']] = df[TEXT_COLUMN].apply(
    lambda x: pd.Series(extract_unit_and_value(x))
)

# --- 3d. Create Boolean Feature for Bullet Points Presence ---
# Simple check for the presence of common bullet point indicators
df['has_bullet_points'] = df[TEXT_COLUMN].str.contains(r'Bullet Point', regex=False).astype(int)

# --- 3e. BRAND EXTRACTION (Heuristic) ---
# Simple heuristic: often the second word or phrase after a keyword
def extract_brand(text):
    for keyword in BRAND_KEYWORDS:
        match = re.search(f'{keyword}:\s*([A-Za-z0-9\s,&]+)', text, re.IGNORECASE)
        if match:
            # Take the first few words and clean them
            brand = match.group(1).split(',')[0].strip()
            return brand
    return 'Unknown'

df['brand'] = df[TEXT_COLUMN].apply(extract_brand)


# --- 4. TF-IDF VECTORIZATION on Cleaned Text ---
print(f"\nStep 4: Creating TF-IDF vectors (Components: {SVD_COMPONENTS})...")

# 4a. Fit and transform the text data
tfidf_vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=MIN_DF
)

tfidf_matrix = tfidf_vectorizer.fit_transform(tqdm(df['cleaned_content'], desc="TF-IDF Fit/Transform"))
print(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")

# 4b. Dimensionality Reduction with Truncated SVD
svd_model = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=42)
text_embeddings = svd_model.fit_transform(tfidf_matrix)
print(f"SVD Embeddings shape: {text_embeddings.shape}")


# --- 5. Finalize and Save the Engineered/Text Features ---
print("\nStep 5: Saving all text and engineered features...")

# Create DataFrame for SVD Embeddings
feature_cols = [f'text_feat_{i}' for i in range(SVD_COMPONENTS)]
df_text_features = pd.DataFrame(text_embeddings, columns=feature_cols)

# Combine Embeddings with Engineered Features and Sample ID
df_final = pd.concat([
    df[['sample_id', 'product_name', 'unit', 'unit_value', 'brand', 'has_bullet_points']],
    df_text_features
], axis=1)

# Clean up engineered features
df_final['unit'] = df_final['unit'].fillna('other').astype('category')
df_final['brand'] = df_final['brand'].fillna('Unknown').astype('category')

# Drop helper columns and save
df_final.to_csv(OUTPUT_FEATURES_PATH, index=False)

print(f"\n--- SUCCESS ---")
print(f"All features saved to {OUTPUT_FEATURES_PATH}")
print(f"File contains {df_final.shape[0]} rows and {df_final.shape[1]} columns.")
print("Next Step: Merge this file, the embeds_image.csv, and the target variable for LightGBM training!")