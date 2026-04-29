import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
import sys
import os
import warnings
warnings.filterwarnings("ignore") 

# --- CONFIGURATION ---
INPUT_ENGINEERED_PATH = 'data_intermediates/test_intermediate_engineered_features.csv'
OUTPUT_FEATURES_PATH = 'data_intermediates/test_embeds_text_engineered.csv'
ASSETS_PATH = 'assets_trained/trained_text_assets.pkl'
TEXT_COLUMN_CLEAN = 'cleaned_content'

# --- Load Test Engineered Data ---
print(f"Loading test engineered data from {INPUT_ENGINEERED_PATH}...")
try:
    df_test = pd.read_csv(INPUT_ENGINEERED_PATH)
except FileNotFoundError:
    print(f"Error: File not found at {INPUT_ENGINEERED_PATH}")
    sys.exit(1)

text_data = df_test[TEXT_COLUMN_CLEAN].fillna('')

# --- Load Trained TF-IDF and SVD Assets ---
with open(ASSETS_PATH, 'rb') as f:
    trained_assets = pickle.load(f)

tfidf_vectorizer = trained_assets['tfidf_vectorizer']
svd_model = trained_assets['svd_model']

# --- Transform Text to Embeddings ---
print("Transforming test text to TF-IDF vectors...")
tfidf_matrix = tfidf_vectorizer.transform(tqdm(text_data, desc="TF-IDF Transform"))
print("Reducing dimensions using SVD...")
text_embeddings = svd_model.transform(tfidf_matrix)

# --- Save Test Embeddings ---
feature_cols = [f'text_feat_{i}' for i in range(text_embeddings.shape[1])]
df_text_features = pd.DataFrame(text_embeddings, columns=feature_cols)
df_text_output = pd.concat([df_test[['sample_id']], df_text_features], axis=1)

os.makedirs(os.path.dirname(OUTPUT_FEATURES_PATH), exist_ok=True)
df_text_output.to_csv(OUTPUT_FEATURES_PATH, index=False)
print(f"Test text embeddings saved to {OUTPUT_FEATURES_PATH}")
