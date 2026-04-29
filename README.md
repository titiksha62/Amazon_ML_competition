# Amazon ML Competition: E-commerce Price Predictor

## Overview
This repository contains the solution for predicting product prices using a multi-modal machine learning approach. The models utilize both text (catalog descriptions, bullet points, etc.) and images to accurately forecast the price of an e-commerce product.

**Core Technologies:**
- **LightGBM Ensemble:** For tabular and vectorized text features.
- **EfficientNetB0 (TensorFlow/Keras):** For extracting image embeddings.
- **TF-IDF & Truncated SVD:** For processing catalog text and generating description embeddings.
- **Streamlit:** For the interactive web demo.

## Project Structure
To keep the repository clean and maintainable, the codebase is organized into the following structure:
- `app2.py` - The main, interactive Streamlit demo application.
- `pipeline/` - Contains the sequential data processing, feature engineering, and model training scripts (`01_engineer_text.py` through `06_predict_and_submit.py`).
- `utils/` - Contains standalone data utility scripts (`feature_extraction.py`, `extract_image_embed.py`, etc.).
- `Code_files/` - Contains legacy and backup scripts (including the old `app.py`).
- `data_raw/`, `data_intermediates/`, `assets_trained/` - Local directories for data and model storage (ignored by Git due to their size).

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/titiksha62/Amazon_ML_competition.git
   cd Amazon_ML_competition
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Generating the Pre-trained Models

> **⚠️ Note on Pre-trained Models:** To keep the repository lightweight, the large trained models (`assets_trained/`) and raw datasets (`data_raw/`) are excluded from Git. You **must** generate these models locally using the provided pipeline scripts before running the demo.

### Heavyweight Training Steps
To generate the necessary model assets (`trained_lgbm_models_2.pkl` and `trained_text_assets.pkl`), you must execute the pipeline from the **root directory**.

1. **Prepare Data**: Ensure your raw dataset (e.g., `train_image_id.csv`) is placed inside the `data_raw/` directory.
2. **Feature Engineering**: Run the text engineering script to clean and format the catalog data.
   ```bash
   python pipeline/01_engineer_text.py
   ```
3. **Generate Text Embeddings**: Create the TF-IDF and SVD representations.
   ```bash
   python pipeline/02_embed_text_and_save.py
   ```
4. **Prepare Image Data & Merge**: Extract image embeddings and merge them with the text features.
   ```bash
   python pipeline/03_a_prepare_data.py
   python pipeline/03_b_merge_features.py
   ```
5. **Train Models**: Finally, train the LightGBM models. The required `.pkl` files will automatically be saved into the `assets_trained/` directory.
   ```bash
   python pipeline/04_train_model_and_save.py
   ```

## Running the Demo (`app2.py`)
Once the models are successfully generated and saved in `assets_trained/`:

1. Launch the Streamlit application from the root directory:
   ```bash
   streamlit run app2.py
   ```
2. Open the URL provided by Streamlit in your browser (usually `http://localhost:8501`).
3. **Upload an Image:** Use the uploader on the left to add a product image (JPG/PNG).
4. **Enter Description:** Paste the product's catalog details in the text area on the right.
5. Click **"🔮 Predict Price"** to generate the multi-modal price prediction!