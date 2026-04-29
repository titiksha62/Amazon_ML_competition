# Amazon ML Competition: E-commerce Price Predictor

## Overview
This repository contains the solution for predicting product prices using a multi-modal machine learning approach. The models utilize both text (catalog descriptions, bullet points, etc.) and images to accurately forecast the price of an e-commerce product.

**Core Technologies:**
- **LightGBM Ensemble:** For tabular and vectorized text features.
- **EfficientNetB0 (TensorFlow/Keras):** For extracting image embeddings.
- **TF-IDF & Truncated SVD:** For processing catalog text and generating description embeddings.
- **Streamlit:** For the interactive web demo.

## Project Structure
- `app2.py` - The main, interactive Streamlit demo application.
- `app.py` - An alternative/legacy version of the Streamlit demo.
- `01_engineer_text.py` through `06_predict_and_submit.py` - The end-to-end data processing, feature engineering, model training, and submission generation pipeline.
- `feature_extraction.py`, `extract_image_embed.py`, `generate_text_embeddings.py` - Utilities for extracting features from the raw datasets.

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

## Running the Demo (`app2.py`)

> **⚠️ Note on Pre-trained Models:** To keep the repository lightweight, the large trained models (`assets_trained/` directory) and the raw CSV datasets are excluded from Git. You **must** generate these models locally using the provided pipeline scripts (e.g., `04_train_model_and_save.py`) before running the demo. Ensure that `trained_lgbm_models_2.pkl` and `trained_text_assets.pkl` are present in your `assets_trained/` folder.

Once the models are saved in the appropriate directory:

1. Launch the Streamlit application:
   ```bash
   streamlit run app2.py
   ```
2. Open the URL provided by Streamlit in your browser (usually `http://localhost:8501`).
3. **Upload an Image:** Use the uploader on the left to add a product image (JPG/PNG).
4. **Enter Description:** Paste the product's catalog details in the text area on the right.
5. Click **"🔮 Predict Price"** to generate the multi-modal price prediction!