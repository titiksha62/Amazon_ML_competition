"""
Feature Extraction for Depression Detection
Extracts audio, visual, gaze, and pose features from DAIC-WOZ dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def get_severity(score):
    """Convert PHQ-8 score to severity level (0-4)"""
    if score <= 4:
        return 0  # None-minimal
    elif score <= 9:
        return 1  # Mild
    elif score <= 14:
        return 2  # Moderate
    elif score <= 19:
        return 3  # Moderately Severe
    else:
        return 4  # Severe

def safe_read_csv(file_path, sep=','):
    """Safely read CSV file with error handling"""
    try:
        return pd.read_csv(file_path, sep=sep, on_bad_lines='skip')
    except Exception as e:
        print(f"  ⚠️ Error reading {file_path.name}: {e}")
        return None

def extract_audio_features(session_folder, session_id):
    """Extract audio features from COVAREP file"""
    features = {}
    covarep_file = session_folder / f"{session_id}_COVAREP.csv"
    
    if not covarep_file.exists():
        return features
    
    df = safe_read_csv(covarep_file)
    if df is None or len(df) == 0:
        return features
    
    # Remove scrubbed entries (all zeros)
    df = df[(df != 0).any(axis=1)]
    
    if len(df) == 0:
        return features
    
    # Key audio features
    audio_cols = ['F0', 'VUV', 'NAQ', 'QOQ', 'H1H2', 'PSP', 'MDQ', 'peakSlope', 'Rd']
    
    for col in audio_cols:
        if col in df.columns:
            values = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) > 0:
                features[f'audio_{col}_mean'] = values.mean()
                features[f'audio_{col}_std'] = values.std()
                features[f'audio_{col}_max'] = values.max()
                features[f'audio_{col}_min'] = values.min()
    
    # MFCC features (first 13)
    mfcc_cols = [f'MCEP_{i}' for i in range(13)]
    for col in mfcc_cols:
        if col in df.columns:
            values = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) > 0:
                features[f'audio_{col}_mean'] = values.mean()
                features[f'audio_{col}_std'] = values.std()
    
    return features

def extract_visual_features(session_folder, session_id):
    """Extract facial action unit features"""
    features = {}
    au_file = session_folder / f"{session_id}_CLNF_AUs.csv"
    
    if not au_file.exists():
        return features
    
    df = safe_read_csv(au_file)
    if df is None or len(df) == 0:
        return features
    
    # Only use successful detections
    df = df[df['success'] == 1]
    
    if len(df) == 0:
        return features
    
    # Action Unit regression (intensity)
    au_regression = [col for col in df.columns if col.endswith('_r')]
    for col in au_regression:
        values = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) > 0:
            features[f'visual_{col}_mean'] = values.mean()
            features[f'visual_{col}_std'] = values.std()
            features[f'visual_{col}_max'] = values.max()
    
    # Action Unit classification (presence)
    au_binary = [col for col in df.columns if col.endswith('_c')]
    for col in au_binary:
        features[f'visual_{col}_activation_rate'] = df[col].mean()
    
    return features

def extract_gaze_features(session_folder, session_id):
    """Extract gaze direction features"""
    features = {}
    gaze_file = session_folder / f"{session_id}_CLNF_gaze.txt"
    
    if not gaze_file.exists():
        return features
    
    df = safe_read_csv(gaze_file)
    if df is None or len(df) == 0:
        return features
    
    # Clean column names (remove spaces)
    df.columns = df.columns.str.strip()
    
    df = df[df['success'] == 1]
    
    if len(df) == 0:
        return features
    
    # Gaze direction coordinates
    gaze_cols = ['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1']
    for col in gaze_cols:
        if col in df.columns:
            values = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) > 0:
                features[f'gaze_{col}_mean'] = values.mean()
                features[f'gaze_{col}_std'] = values.std()
    
    return features

def extract_pose_features(session_folder, session_id):
    """Extract head pose features"""
    features = {}
    pose_file = session_folder / f"{session_id}_CLNF_pose.txt"
    
    if not pose_file.exists():
        return features
    
    df = safe_read_csv(pose_file)
    if df is None or len(df) == 0:
        return features
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    df = df[df['success'] == 1]
    
    if len(df) == 0:
        return features
    
    # Head rotation
    pose_cols = ['Rx', 'Ry', 'Rz', 'X', 'Y', 'Z']
    for col in pose_cols:
        if col in df.columns:
            values = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) > 0:
                features[f'pose_{col}_mean'] = values.mean()
                features[f'pose_{col}_std'] = values.std()
                features[f'pose_{col}_range'] = values.max() - values