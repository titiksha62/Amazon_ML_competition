# app.py
# ==========================================
# E-commerce Price Prediction Streamlit App
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import os
import tempfile
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# ML Imports
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------------------------
# IMPORTANT: must be the first Streamlit command
# -------------------------
st.set_page_config(page_title="E-commerce Price Predictor", layout="wide")



# ==========================================
# CONFIGURATION
# ==========================================
ASSETS_PATH_LGBM = 'assets_trained/trained_lgbm_models_2.pkl'
ASSETS_PATH_TEXT = 'assets_trained/trained_text_assets.pkl'
IMAGE_SIZE = 224  # For EfficientNetB0

# --- CONSTANTS (shared with training) ---
COMMON_UNITS_REGEX = r'(\d+\.?\d*)\s*[-]*\s*(oz|fl oz|lb|kg|g|mg|ml|l|count|pack|ct|pair|set|unit|sheets|bags|ounce|pound)'
CONVERSION_FACTORS = {
    'lb': 16.0, 'kg': 35.274, 'g': 0.035274, 'mg': 0.000035,
    'ounce': 1.0, 'oz': 1.0, 'pound': 16.0,
    'l': 33.814, 'ml': 0.033814, 'fl oz': 1.0,
    'count': 1.0, 'pack': 1.0, 'ct': 1.0, 'pair': 1.0, 'set': 1.0,
    'unit': 1.0, 'sheets': 1.0, 'bags': 1.0
}
WEIGHT_UNITS = ['oz', 'ounce', 'lb', 'kg', 'g', 'mg', 'pound']
VOLUME_UNITS = ['fl oz', 'ml', 'l']
PRODUCT_STARTERS = ['The', 'A', 'Best', 'Original', 'Deluxe', 'Premium', 'New', 'Fresh', 'Simply', 'Good']

# ==========================================
# FEATURE ENGINEERING UTILITIES
# ==========================================
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
    if not product_name:
        return 'Unknown'
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
    if pd.isna(unit):
        return None
    unit = str(unit).lower().strip().replace('.', '')
    if unit in ['ounces', 'ozs']:
        return 'oz'
    if unit in ['pounds', 'lbs']:
        return 'lb'
    if unit in ['liters', 'lts']:
        return 'l'
    if unit in ['counts', 'cts']:
        return 'count'
    if unit in ['packs', 'pks']:
        return 'pack'
    return unit

def extract_and_standardize_unit(text):
    match = re.search(COMMON_UNITS_REGEX, text.lower())
    if not match:
        return np.nan, 'other'
    try:
        value = float(match.group(1))
    except Exception:
        return np.nan, 'other'
    raw_unit = match.group(2)
    unit = clean_unit_string(raw_unit)
    unit_type = 'other'
    unit_value_std = np.nan

    if unit in WEIGHT_UNITS:
        unit_type = 'weight'
    elif unit in VOLUME_UNITS:
        unit_type = 'volume'
    elif unit in CONVERSION_FACTORS:
        unit_type = 'count'

    if unit_type == 'weight':
        unit_value_std = value * CONVERSION_FACTORS.get('oz', 1.0)
    elif unit_type == 'volume':
        unit_value_std = value * CONVERSION_FACTORS.get('fl oz', 1.0)
    elif unit_type == 'count':
        unit_value_std = value

    return unit_value_std, unit_type

# ==========================================
# MODEL + ASSETS LOADING
# (cached to avoid reloading on each interaction)
# ==========================================
@st.cache_resource
def load_assets():
    """Load pretrained models and assets for inference."""
    # Load LGBM ensemble
    if not os.path.exists(ASSETS_PATH_LGBM):
        st.error(f"LGBM asset not found at: {ASSETS_PATH_LGBM}")
        st.stop()
    with open(ASSETS_PATH_LGBM, 'rb') as f:
        lgbm_models = pickle.load(f)

    # Load text assets (TF-IDF + SVD)
    if not os.path.exists(ASSETS_PATH_TEXT):
        st.error(f"Text asset not found at: {ASSETS_PATH_TEXT}")
        st.stop()
    with open(ASSETS_PATH_TEXT, 'rb') as f:
        text_assets = pickle.load(f)
    tfidf = text_assets.get('tfidf_vectorizer')
    svd = text_assets.get('svd_model')

    # Load EfficientNetB0 as image feature extractor
    try:
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet', include_top=False, pooling='avg',
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
        )
        base_model.trainable = False
        image_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)
    except Exception as e:
        st.error(f"Failed to initialize EfficientNetB0: {e}")
        st.stop()

    return lgbm_models, tfidf, svd, image_extractor

# Load assets once (cached)
LGBM_MODELS, TFIDF_MODEL, SVD_MODEL, IMAGE_EXTRACTOR = load_assets()

# Validate loaded objects
if not isinstance(LGBM_MODELS, (list, tuple)) or len(LGBM_MODELS) == 0:
    st.error("Loaded LGBM models object is invalid.")
    st.stop()

TRAINED_FEATURES = list(LGBM_MODELS[0].feature_name_)

# ==========================================
# CORE PREDICTION PIPELINE
# ==========================================
def get_single_prediction(catalog_content, uploaded_file_path):
    """Generate a single price prediction for given text + image."""
    # --- Step 1: Basic DF ---
    df = pd.DataFrame({'catalog_content': [catalog_content], 'sample_id': [99999]})
    df['catalog_content'] = df['catalog_content'].fillna('').astype(str)

    # --- Step 2: Manual Features ---
    df['cleaned_content'] = df['catalog_content'].apply(clean_for_embeddings)
    df['product_name'] = df['catalog_content'].apply(extract_product_name).fillna('')
    df['brand'] = df['product_name'].apply(extract_brand_from_name)
    df[['unit_value_std', 'unit_type']] = df['catalog_content'].apply(
        lambda x: pd.Series(extract_and_standardize_unit(x))
    )
    df['has_bullet_points'] = df['catalog_content'].str.contains(r'Bullet Point', regex=False).astype(int)

    # --- Step 3: Text Features (TF-IDF + SVD) ---
    tfidf_matrix = TFIDF_MODEL.transform(df['cleaned_content'])
    text_embeddings = SVD_MODEL.transform(tfidf_matrix)
    df_text_features = pd.DataFrame(text_embeddings, columns=[f'text_feat_{i}' for i in range(text_embeddings.shape[1])])

    # --- Step 4: Image Features ---
    IMG_EMBEDDING_DIM = 1280
    img_embeddings = np.zeros(IMG_EMBEDDING_DIM)
    try:
        img = Image.open(uploaded_file_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.expand_dims(np.array(img, dtype='float32'), axis=0)
        img_input = preprocess_input(img_array)
        img_embeddings = IMAGE_EXTRACTOR.predict(img_input, verbose=0)[0]
    except Exception:
        # safe fallback to zero-vector if image processing fails
        pass
    df_image_features = pd.DataFrame([img_embeddings], columns=[f'img_feat_{i}' for i in range(IMG_EMBEDDING_DIM)])

    # --- Step 5: Merge ---
    df_final = pd.concat([df.reset_index(drop=True),
                          df_text_features.reset_index(drop=True),
                          df_image_features.reset_index(drop=True)], axis=1)
    cols_to_drop = ['catalog_content', 'cleaned_content', 'product_name']
    df_final.drop(columns=[c for c in cols_to_drop if c in df_final.columns], inplace=True, errors='ignore')

    # --- Step 6: Align Columns ---
    X_predict = df_final.reindex(columns=TRAINED_FEATURES, fill_value=0)

    # Convert categorical columns back if present
    for col in ['unit_type', 'brand']:
        if col in X_predict.columns:
            X_predict[col] = X_predict[col].astype('category')

    # --- Step 7: Predict from Ensemble ---
    try:
        all_preds = np.array([model.predict(X_predict)[0] for model in LGBM_MODELS])
    except Exception as e:
        raise RuntimeError(f"Error during LGBM prediction: {e}")

    avg_log_preds = np.mean(all_preds)
    final_price = np.expm1(avg_log_preds)

    return max(0.01, float(final_price))

# ==========================================
# STREAMLIT APP UI
# ==========================================
st.title("💰 E-commerce Price Predictor")
st.markdown("Predict product prices using **multi-modal (image + text)** machine learning models trained with LightGBM and EfficientNet.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🖼️ 1. Upload Product Image")
    uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Product Image", use_container_width=True, width=250)


with col2:
    st.subheader("📝 2. Enter Catalog Description")
    catalog_input = st.text_area(
        "Paste full product text here:",
        height=300,
        placeholder="Example:\nItem Name: Organic Arabica Coffee.\nProduct Details: 12 oz pack. Bullet Point: Freshly roasted beans."
    )

st.divider()

if st.button("🔮 Predict Price", type="primary"):
    if not catalog_input:
        st.error("Please provide catalog content.")
    elif uploaded_file is None:
        st.warning("Please upload a product image.")
    else:
        # Save uploaded file temporarily for Keras/Numpy processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        with st.spinner("Analyzing product and generating price prediction..."):
            try:
                predicted_price = get_single_prediction(catalog_input, tmp_file_path)
                st.success(f"**Predicted Price:** ${predicted_price:,.2f}")
                st.info("This prediction is based on a LightGBM ensemble trained on text, image, and manual features.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
            finally:
                try:
                    os.remove(tmp_file_path)
                except Exception:
                    pass

st.caption("© 2025 Price Prediction AI | Developed using LightGBM + EfficientNet + Streamlit")
