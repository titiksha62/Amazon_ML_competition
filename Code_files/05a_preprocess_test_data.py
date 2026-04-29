import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import sys
import os
import warnings
warnings.filterwarnings("ignore") 

# --- CONFIGURATION ---
# Load the master test file that already includes 'image_id'
MASTER_TEST_DATA_PATH = 'data_raw/test_image_id.csv' 
OUTPUT_ENGINEERED_PATH = 'data_intermediates/test_intermediate_engineered_features.csv'
TEXT_COLUMN = 'catalog_content'

# --- CONSTANTS for Feature Engineering (Must match 01_engineer_text.py) ---
COMMON_UNITS_REGEX = r'(\d+\.?\d*)\s*[-]*\s*(oz|fl oz|lb|kg|g|mg|ml|l|count|pack|ct|pair|set|unit|sheets|bags|ounce|pound)'
CONVERSION_FACTORS = {
    'lb': 16.0, 'kg': 35.274, 'g': 0.035274, 'mg': 0.000035, 'ounce': 1.0, 'oz': 1.0, 'pound': 16.0,
    'l': 33.814, 'ml': 0.033814, 'fl oz': 1.0, 
    'count': 1.0, 'pack': 1.0, 'ct': 1.0, 'pair': 1.0, 'set': 1.0, 'unit': 1.0, 'sheets': 1.0, 'bags': 1.0
}
WEIGHT_UNITS = ['oz', 'ounce', 'lb', 'kg', 'g', 'mg', 'pound']
VOLUME_UNITS = ['fl oz', 'ml', 'l']
PRODUCT_STARTERS = ['The', 'A', 'Best', 'Original', 'Deluxe', 'Premium', 'New', 'Fresh', 'Simply', 'Good']


# --- 2. Utility Functions (Same as 01_engineer_text.py) ---
def clean_for_embeddings(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text) 
    text = re.sub(r'(item name|product name|bullet point|product details|description)', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_product_name(text):
    match = re.search(r'Item Name: (.*)', text)
    if match:
        name = match.group(1).split('.')[0].strip()
        name = re.sub(r'\(.*\)', '', name).strip()
        return name
    return None

def extract_brand_from_name(product_name):
    if not product_name: return 'Unknown'
    words = product_name.split()
    brand_parts = []
    for word in words[:4]:
        if (word and word[0].isupper() and len(word) > 1 and word.isalpha() 
            and word not in PRODUCT_STARTERS):
            brand_parts.append(word)
        else:
            break
    return " ".join(brand_parts) if brand_parts else 'Unknown'

def clean_unit_string(unit): 
    if pd.isna(unit): return None
    unit = str(unit).lower().strip().replace('.', '')
    if unit in ['ounces', 'ozs']: return 'oz'
    if unit in ['pounds', 'lbs']: return 'lb'
    if unit in ['liters', 'lts']: return 'l'
    if unit in ['counts', 'cts']: return 'count'
    if unit in ['packs', 'pks']: return 'pack'
    return unit

def extract_and_standardize_unit(text):
    match = re.search(COMMON_UNITS_REGEX, text.lower())
    if not match: return np.nan, 'other'
    value = float(match.group(1))
    raw_unit = match.group(2)
    unit = clean_unit_string(raw_unit) 
    unit_type = 'other'
    unit_value_std = np.nan
    conversion_factor = CONVERSION_FACTORS.get(unit, np.nan)
    
    if unit in WEIGHT_UNITS: unit_type = 'weight'
    elif unit in VOLUME_UNITS: unit_type = 'volume'
    elif unit in CONVERSION_FACTORS: unit_type = 'count'

    if unit_type == 'weight': unit_value_std = value * CONVERSION_FACTORS['oz']
    elif unit_type == 'volume': unit_value_std = value * CONVERSION_FACTORS['fl oz']
    elif unit_type == 'count': unit_value_std = value
            
    return unit_value_std, unit_type


# --- 3. Load Data and Initial Preprocessing ---
print(f"Step 3: Loading master test data from {MASTER_TEST_DATA_PATH}...")
try:
    df = pd.read_csv(MASTER_TEST_DATA_PATH)
except FileNotFoundError:
    print(f"Error: Master test data CSV not found at {MASTER_TEST_DATA_PATH}. Ensure file is named 'test_image_id.csv' in /data_raw.")
    sys.exit(1)

df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna('').astype(str)


# --- 4. Feature Engineering ---

# 4a. Text Cleaning
df['cleaned_content'] = [clean_for_embeddings(text) for text in tqdm(df[TEXT_COLUMN], desc="Text Cleaning")]

# 4b. Product Name Extraction
df['product_name'] = df[TEXT_COLUMN].apply(extract_product_name).fillna('')

# 4c. Brand Extraction (Manual Feature)
df['brand'] = df['product_name'].apply(extract_brand_from_name)

# 4d. Unit Extraction and Standardization (Manual Features)
df[['unit_value_std', 'unit_type']] = df[TEXT_COLUMN].apply(
    lambda x: pd.Series(extract_and_standardize_unit(x))
)

# 4e. Simple Boolean Feature
df['has_bullet_points'] = df[TEXT_COLUMN].str.contains(r'Bullet Point', regex=False).astype(int)

# 4f. The 'price_per_unit' feature is SKIPPED because we don't have the price (target) on the test set.

# --- 5. Finalize and Save Intermediate Results ---

# Final output selection, keeping all engineered features and necessary identifiers
df_output = df[['sample_id', 'image_id', 'image_link', 
                'unit_value_std', 'unit_type', 
                'brand', 'has_bullet_points', 'cleaned_content']].copy()

# Convert appropriate columns to category type (important for consistency with training)
df_output['unit_type'] = df_output['unit_type'].astype('category')
df_output['brand'] = df_output['brand'].astype('category')

# Create the output directory if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT_ENGINEERED_PATH), exist_ok=True)
df_output.to_csv(OUTPUT_ENGINEERED_PATH, index=False)

print(f"\n--- SUCCESS: Step 05a Complete (Test Feature Engineering) ---")
print(f"Engineered features saved to {OUTPUT_ENGINEERED_PATH}")