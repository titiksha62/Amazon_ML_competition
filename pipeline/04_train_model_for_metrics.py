import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import sys
import os
import warnings

warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
FINAL_TRAIN_PATH = 'data_intermediates/master_train_data_2.csv'
TARGET_COL = 'log_price'
N_SPLITS = 5  # Number of folds for Cross-Validation
OUTPUT_MODELS_PATH = 'assets_trained/trained_lgbm_models_3.pkl'

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

# Categorical Features
CAT_FEATURES = ['unit_type', 'brand']

# Minimum predicted price to avoid negative/zero predictions
MIN_PREDICTED_PRICE = 0.01

# --- 2. Custom Evaluation Metric: SMAPE ---
def smape_metric(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (SMAPE)"""
    y_true_price = np.expm1(y_true)
    y_pred_price = np.expm1(y_pred)
    numerator = np.abs(y_pred_price - y_true_price)
    denominator = (np.abs(y_true_price) + np.abs(y_pred_price)) / 2
    smape_val = np.mean(numerator / np.where(denominator == 0, 1, denominator)) * 100
    return 'SMAPE', smape_val, False

# --- 3. Load Data and Prepare Features ---
print("Step 3: Loading Master Data and defining X and Y...")
try:
    df_master = pd.read_csv(FINAL_TRAIN_PATH)
except FileNotFoundError:
    print(f"Error: Master data CSV not found at {FINAL_TRAIN_PATH}.")
    sys.exit(1)

# Prepare features and target
non_feature_cols = ['sample_id', 'image_id', 'price', 'image_link', TARGET_COL]
X = df_master.drop(columns=[col for col in non_feature_cols if col in df_master.columns])
Y = df_master[TARGET_COL]

# Convert categorical features
for col in CAT_FEATURES:
    if col in X.columns:
        X[col] = X[col].astype('category')

print(f"Features (X) Shape: {X.shape}")

# --- 4. K-Fold Cross-Validation and Training ---
print(f"\nStep 4: Training LightGBM Model with {N_SPLITS}-Fold Cross-Validation...")

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
cv_smape_scores = []
models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, Y)):
    print(f"\n--- FOLD {fold+1}/{N_SPLITS} ---")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    Y_train, Y_val = Y.iloc[train_idx], Y.iloc[val_idx]

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_train, Y_train,
        eval_set=[(X_val, Y_val)],
        eval_metric=lambda y_true, y_pred: [smape_metric(y_true, y_pred)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
        categorical_feature=CAT_FEATURES
    )

    val_preds = model.predict(X_val)
    oof_preds[val_idx] = val_preds

    _, fold_smape, _ = smape_metric(Y_val, val_preds)
    cv_smape_scores.append(fold_smape)
    print(f"Fold {fold+1} Validation SMAPE: {fold_smape:.4f}%")

    models.append(model)

# --- 5. Final Evaluation ---
print("\n--- FINAL EVALUATION ---")
_, overall_smape, _ = smape_metric(Y, oof_preds)
print(f"Mean CV SMAPE: {np.mean(cv_smape_scores):.4f}%")
print(f"Overall OOF SMAPE: {overall_smape:.4f}%")

# Convert back to price scale for other metrics
y_true_price = np.expm1(Y)
y_pred_price = np.expm1(oof_preds)

r2 = r2_score(y_true_price, y_pred_price)
mae = mean_absolute_error(y_true_price, y_pred_price)
rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))

print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# --- 6. Save Trained Models ---
os.makedirs(os.path.dirname(OUTPUT_MODELS_PATH), exist_ok=True)
with open(OUTPUT_MODELS_PATH, 'wb') as f:
    pickle.dump(models, f)

print(f"\n--- SUCCESS: Trained models saved to {OUTPUT_MODELS_PATH} ---")
