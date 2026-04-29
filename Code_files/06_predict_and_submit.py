import pandas as pd
import numpy as np
import pickle
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
INPUT_MASTER_TEST_PATH = 'data_intermediates/master_test_features.csv'
MODELS_ASSETS_PATH = 'assets_trained/trained_lgbm_models.pkl'
SUBMISSION_OUTPUT_PATH = 'submissions/submission.csv'
N_SPLITS = 5 
TARGET_COL = 'log_price'
CAT_FEATURES = ['unit_type', 'brand'] # Define categories here for alignment

# --- 2. Load Data and Trained Models ---
print("Step 2: Loading master test features and trained LGBM models...")

try:
    df_test = pd.read_csv(INPUT_MASTER_TEST_PATH)
    
    with open(MODELS_ASSETS_PATH, 'rb') as f:
        models = pickle.load(f)
    
except FileNotFoundError as e:
    print(f"Error: Required file not found: {e}")
    sys.exit(1)

# --- 3. Prepare Test Data (X_test) and Align Features (CRITICAL FIX) ---
print("Step 3: Preparing and aligning test features for prediction...")

# --- 3a. Drop non-feature columns ---
# Keep only features that the model expects (dropping identifiers)
non_feature_cols = ['sample_id', 'image_id', 'image_link']
X_test = df_test.drop(columns=[col for col in non_feature_cols if col in df_test.columns])

# --- 3b. Align Categorical Columns ---
for col in CAT_FEATURES:
    if col in X_test.columns:
        X_test[col] = X_test[col].astype('category') 
        
# --- 3c. Extract Trained Feature Names ---
# Get the exact feature names the model was trained on from the first model
# This guarantees correct column count and order.
trained_feature_names = list(models[0].feature_name_)
X_test_aligned = X_test.reindex(columns=trained_feature_names, fill_value=0)

print(f"Test Features (X_test) Aligned Shape: {X_test_aligned.shape}")


# --- 4. Generate Predictions ---
print("\nStep 4: Generating ensemble predictions...")

# Initialize prediction array
all_test_preds = np.zeros((len(X_test_aligned), N_SPLITS))

# Predict using each model in the ensemble
for i, model in enumerate(models):
    # Use the aligned test data
    all_test_preds[:, i] = model.predict(X_test_aligned) 
    print(f"Prediction complete for Model {i+1}/{N_SPLITS}")

# Average the predictions
avg_log_preds = all_test_preds.mean(axis=1)


# --- 5. Inverse Transform and Final Formatting ---
print("\nStep 5: Inverse transforming predictions and formatting submission...")

final_prices = np.expm1(avg_log_preds)
final_prices = np.maximum(0, final_prices) 

# Create submission DataFrame
df_submission = pd.DataFrame({
    'sample_id': df_test['sample_id'],
    'price': final_prices
})

# --- 6. Save Submission File ---
os.makedirs(os.path.dirname(SUBMISSION_OUTPUT_PATH), exist_ok=True)
df_submission.to_csv(SUBMISSION_OUTPUT_PATH, index=False)

print(f"\n--- SUCCESS: Step 06 Complete ---")
print(f"Final predictions saved to {SUBMISSION_OUTPUT_PATH}")
print("\nPROJECT PIPELINE COMPLETE! ✅")