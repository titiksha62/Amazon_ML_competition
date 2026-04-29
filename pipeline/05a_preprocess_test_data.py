import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import os
import tempfile
from PIL import Image
from tqdm import tqdm # Although not strictly needed for single prediction, good for consistency
import sys
import warnings
warnings.filterwarnings("ignore")

# Import necessary ML libraries (ensure these are in your requirements.txt)
import lightgbm as lgb 
# Only import if you need to use TensorFlow/Keras specific types, otherwise use tf.keras
try:
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input
except ImportError:
    st.error("TensorFlow not found. Please ensure it is installed to run the image model.")
    sys.exit()

# --- 1. CONFIGURATION AND CONSTANTS ---
ASSETS_PATH_LGBM = 'assets_trained/trained_lgbm_models.pkl'
ASSETS_PATH_TEXT = 'assets_trained/trained_text_assets.pkl'
IMAGE_SIZE = 224 # EfficientNet size

# CONSTANTS for Feature Engineering (Copied directly from 05a)
COMMON_UNITS_REGEX = r'(\d+\.?\d*)\s*[-]*\s*(oz|fl oz|lb|kg|g|mg|ml|l|count|pack|ct|pair|set|unit|sheets|bags|ounce|pound)'
CONVERSION_FACTORS = {
    'lb': 16.0, 'kg': 35.274, 'g': 0.035274, 'mg': 0.000035, 'ounce': 1.0, 'oz': 1.0, 'pound': 16.0,
    'l': 33.814, 'ml': 0.033814, 'fl oz': 1.0, 
    'count': 1.0, 'pack': 1.0, 'ct': 1.0, 'pair': 1.0, 'set': 1.0, 'unit': 1.0, 'sheets': 1.0, 'bags': 1.0
}
WEIGHT_UNITS = ['oz', 'ounce', 'lb', 'kg', 'g', 'mg', 'pound']
VOLUME_UNITS = ['fl oz', 'ml', 'l']
PRODUCT_STARTERS = ['The', 'A', 'Best', 'Original', 'Deluxe', 'Premium', 'New', 'Fresh', 'Simply', 'Good']


# --- 2. FEATURE ENGINEERING UTILITY FUNCTIONS (Copied from 05a) ---

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
        # Using the global PRODUCT_STARTERS list
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
    # Using the global COMMON_UNITS_REGEX
    match = re.search(COMMON_UNITS_REGEX, text.lower())
    if not match: return np.nan, 'other'
    value = float(match.group(1))
    raw_unit = match.group(2)
    unit = clean_unit_string(raw_unit) 
    unit_type = 'other'
    unit_value_std = np.nan
    
    # Using the global lists and dicts
    if unit in WEIGHT_UNITS: unit_type = 'weight'
    elif unit in VOLUME_UNITS: unit_type = 'volume'
    elif unit in CONVERSION_FACTORS: unit_type = 'count'

    if unit_type == 'weight': unit_value_std = value * CONVERSION_FACTORS['oz']
    elif unit_type == 'volume': unit_value_std = value * CONVERSION_FACTORS['fl oz']
    elif unit_type == 'count': unit_value_std = value
            
    return unit_value_std, unit_type


# --- 3. ASSET LOADING (Cached) ---

@st.cache_resource
def load_assets():
    """Loads all pre-trained models and assets."""
    # Load LGBM Ensemble
    try:
        with open(ASSETS_PATH_LGBM, 'rb') as f:
            lgbm_models = pickle.load(f)
    except FileNotFoundError:
        st.error(f"LGBM models not found at {ASSETS_PATH_LGBM}. Did you run 04_train_model_and_save.py?")
        sys.exit()

    # Load Text Assets (TFIDF and SVD)
    try:
        with open(ASSETS_PATH_TEXT, 'rb') as f:
            text_assets = pickle.load(f)
        tfidf = text_assets['tfidf_vectorizer']
        svd = text_assets['svd_model']
    except FileNotFoundError:
        st.error(f"Text assets not found at {ASSETS_PATH_TEXT}. Did you run 02_embed_text_features.py?")
        sys.exit()

    # Load Image Model (EfficientNetB0)
    base_model = tf.keras.applications.EfficientNetB0(
        weights='imagenet', include_top=False, pooling='avg', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    base_model.trainable = False
    image_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)
    
    return lgbm_models, tfidf, svd, image_extractor

LGBM_MODELS, TFIDF_MODEL, SVD_MODEL, IMAGE_EXTRACTOR = load_assets()
# Get the list of feature names from the first trained model for alignment
TRAINED_FEATURES = list(LGBM_MODELS[0].feature_name_)


# --- 4. CORE PREDICTION FUNCTION ---

def get_single_prediction(catalog_content, uploaded_file_path):
    # a) Create a DataFrame for this single sample
    data = {'catalog_content': [catalog_content], 'sample_id': [99999]}
    df = pd.DataFrame(data)
    df['catalog_content'] = df['catalog_content'].fillna('').astype(str)

    # b) Manual Feature Engineering (using the defined functions)
    df['cleaned_content'] = df['catalog_content'].apply(clean_for_embeddings) 
    df['product_name'] = df['catalog_content'].apply(extract_product_name).fillna('')
    df['brand'] = df['product_name'].apply(extract_brand_from_name)
    df[['unit_value_std', 'unit_type']] = df['catalog_content'].apply(
        lambda x: pd.Series(extract_and_standardize_unit(x))
    )
    df['has_bullet_points'] = df['catalog_content'].str.contains(r'Bullet Point', regex=False).astype(int)
    
    # c) Text Embedding (Use loaded TFIDF and SVD models to .transform())
    tfidf_matrix = TFIDF_MODEL.transform(df['cleaned_content'])
    text_embeddings = SVD_MODEL.transform(tfidf_matrix)
    df_text_features = pd.DataFrame(text_embeddings, columns=[f'text_feat_{i}' for i in range(text_embeddings.shape[1])])

    # d) Image Embedding (Use loaded Keras model)
    IMG_EMBEDDING_DIM = 1280
    img_embeddings = np.zeros(IMG_EMBEDDING_DIM) # Default to zero vector
    try:
        # Load, resize, and preprocess image
        img = Image.open(uploaded_file_path).resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.expand_dims(np.array(img, dtype='float32'), axis=0)
        img_input = preprocess_input(img_array)
        
        # Predict features
        img_embeddings = IMAGE_EXTRACTOR.predict(img_input, verbose=0)[0]
    except Exception:
        # If image processing fails, the zero vector imputation remains
        pass 

    df_image_features = pd.DataFrame([img_embeddings], columns=[f'img_feat_{i}' for i in range(IMG_EMBEDDING_DIM)])

    # e) Merge and Align
    # Drop intermediate columns before merging
    cols_to_drop = ['catalog_content', 'cleaned_content', 'product_name']
    df_base = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    df_final = pd.concat([df_base.reset_index(drop=True), 
                          df_text_features.reset_index(drop=True), 
                          df_image_features.reset_index(drop=True)], axis=1)

    # f) Align and Predict (CRITICAL ALIGNMENT)
    # Reindex to ensure feature order matches the trained model
    X_predict = df_final.reindex(columns=TRAINED_FEATURES, fill_value=0)
    
    # Convert categorical columns back (LGBM requirement)
    for col in ['unit_type', 'brand']:
         if col in X_predict.columns:
             X_predict[col] = X_predict[col].astype('category')
             
    # Average predictions from the ensemble (outputs log_price)
    all_preds = np.array([model.predict(X_predict)[0] for model in LGBM_MODELS])
    avg_log_preds = np.mean(all_preds)
    
    # Inverse Transform: Price = e^(log_price) - 1
    final_price = np.expm1(avg_log_preds)
    
    return max(0.01, final_price)

# --- 5. STREAMLIT UI ---

st.title("💰 E-commerce Price Predictor")
st.markdown("Upload a product image and provide its catalog description to predict the log-price using our multi-modal ensemble model.")

# --- Input Fields ---
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Product Image")
        uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        
    with col2:
        st.subheader("2. Catalog Content (Text)")
        catalog_input = st.text_area("Paste Full Catalog Text Here:", height=300, 
                                     placeholder="Example: Item Name: Best Organic Coffee. Product Details: 12 oz weight. Bullet Point: Premium blend.")

# --- Prediction Button and Logic ---
if st.button("Predict Price", type="primary"):
    if not catalog_input:
        st.error("Please provide the catalog content.")
    elif uploaded_file is None:
        st.warning("Please upload a product image.")
    else:
        # Save uploaded file temporarily for Keras/Numpy processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        with st.spinner('Generating features and predicting...'):
            try:
                predicted_price = get_single_prediction(catalog_input, tmp_file_path)

                st.markdown("---")
                st.subheader("Prediction Result")
                st.success(f"The predicted price is: **${predicted_price:,.2f}**")
                st.info("This prediction is powered by an ensemble of LightGBM models trained on text, image, and manual features.")
            except Exception as e:
                st.error(f"An error occurred during prediction. Please check the console logs for details. Error: {e}")
            finally:
                # Clean up the temporary file
                os.remove(tmp_file_path)