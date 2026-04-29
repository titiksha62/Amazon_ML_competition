import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
import pickle  # CRITICAL: For saving the trained ensemble models
import sys
import os
import warnings

warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
FINAL_TRAIN_PATH = 'data_intermediates/master_train_data_2.csv'
TARGET_COL = 'log_price'
N_SPLITS = 5  # Number of folds for Cross-Validation
# CRITICAL NEW PATH: Save models to /assets_trained
OUTPUT_MODELS_PATH = 'assets_trained/trained_lgbm_models_2.pkl'

# LightGBM Hyperparameters
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 3000,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt',
}

# --- 2. Custom Evaluation Metric (SMAPE) ---
def smape_metric(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (SMAPE)"""
    # Convert log-predictions back to actual price scale
    y_true_price = np.expm1(y_true)
    y_pred_price = np.expm1(y_pred)
    
    numerator = np.abs(y_pred_price - y_true_price)
    denominator = (np.abs(y_true_price) + np.abs(y_pred_price)) / 2
    
    # Handle division by zero
    smape_val = np.mean(numerator / np.where(denominator == 0, 1, denominator)) * 100
    
    # LightGBM expects (metric_name, metric_value, higher_is_better)
    return 'SMAPE', smape_val, False

# --- 3. Load Data and Prepare Features ---
print("Step 3: Loading Master Data and defining X and Y...")

try:
    df_master = pd.read_csv(FINAL_TRAIN_PATH)
except FileNotFoundError:
    print(f"Error: Master data CSV not found at {FINAL_TRAIN_PATH}. Run 03_a_merge_features.py first.")
    sys.exit(1)

# Features (X): Drop identifiers, the target variable, and the redundant 'price'
non_feature_cols = ['sample_id', 'image_id', 'price', 'image_link', TARGET_COL]
X = df_master.drop(columns=[col for col in non_feature_cols if col in df_master.columns])
Y = df_master[TARGET_COL]

# Identify categorical features for LightGBM
CAT_FEATURES = ['unit_type', 'brand']
for col in CAT_FEATURES:
    if col in X.columns:
        # Important: Convert to 'category' type for LightGBM efficiency
        X[col] = X[col].astype('category')

print(f"Features (X) Shape: {X.shape}")

# --- 4. K-Fold Cross-Validation and Training ---
print(f"\nStep 4: Training LightGBM Model with {N_SPLITS}-Fold Cross-Validation...")

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))  # Out-Of-Fold predictions
cv_smape_scores = []
models = []  # List to store the 5 trained models

for fold, (train_index, val_index) in enumerate(kf.split(X, Y)):
    print(f"\n--- FOLD {fold+1}/{N_SPLITS} ---")
    
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    Y_train, Y_val = Y.iloc[train_index], Y.iloc[val_index]

    model = lgb.LGBMRegressor(**LGBM_PARAMS)

    model.fit(
        X_train, Y_train,
        eval_set=[(X_val, Y_val)],
        eval_metric=lambda y_true, y_pred: [smape_metric(y_true, y_pred)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
        categorical_feature=CAT_FEATURES
    )

    val_preds = model.predict(X_val)
    oof_preds[val_index] = val_preds
    
    _, fold_smape, _ = smape_metric(Y_val, val_preds)
    cv_smape_scores.append(fold_smape)
    
    print(f"Fold {fold+1} Validation SMAPE: {fold_smape:.4f}%")
    
    models.append(model)  # Save the trained model instance

# --- 5. Final Evaluation and Saving ---
print("\n--- FINAL RESULTS ---")

# FIX APPLIED HERE: Correctly unpack the tuple returned by smape_metric
_, overall_smape, _ = smape_metric(Y, oof_preds)

print(f"Mean CV SMAPE across all folds: {np.mean(cv_smape_scores):.4f}%")
print(f"Overall Out-Of-Fold SMAPE: {overall_smape:.4f}%")  # This line now works!

# 🚨 CRUCIAL STEP: Save the list of trained models
os.makedirs(os.path.dirname(OUTPUT_MODELS_PATH), exist_ok=True)
with open(OUTPUT_MODELS_PATH, 'wb') as f:
    pickle.dump(models, f)

print(f"\n--- SUCCESS: Step 4 Complete ---")
print(f"Trained ensemble models saved to {OUTPUT_MODELS_PATH}")
