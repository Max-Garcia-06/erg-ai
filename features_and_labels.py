# features_and_labels.py
"""
Feature extraction and label generation for rowing ergometer data.
Creates windowed features for ML model training.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURATION
# =====================================================================

# Input/Output paths
INPUT_PATHS = glob.glob("sample_data/*.csv")
OUT_DIR = Path("data_processed")
OUT_DIR.mkdir(exist_ok=True)

# Window settings (must match inference)
WINDOW_SIZE = 30
STEP_SIZE = 5
MIN_FILE_LENGTH = 50  # Minimum rows required

# Feature thresholds
WORK_THRESHOLD_PERCENTILE = 50  # Median watts for work/rest classification

# Normalization ranges (for stroke quality scoring)
EFFICIENCY_RANGE = (0.8, 2.5)  # watts/stroke_rate typical range
CONSISTENCY_RANGE = (5, 50)    # rolling std range
SMOOTHNESS_RANGE = (5, 50)     # jitter range


# =====================================================================
# DATA LOADING & NORMALIZATION
# =====================================================================

def load_and_normalize(path: str) -> pd.DataFrame:
    """
    Load CSV file and normalize column names.
    
    Args:
        path: Path to CSV file
        
    Returns:
        DataFrame with normalized columns
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Failed to read {path}: {e}")
    
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Rename common variations
    df.rename(columns={
        'time (s)': 'time',
        'time (seconds)': 'time',
        'time(s)': 'time',
        'pace (sec/500m)': 'pace',
        'pace (seconds)': 'pace',
        'watts': 'watts',
        'power': 'watts',
        'power (watts)': 'watts',
        'stroke rate': 'stroke_rate',
        'stroke_rate': 'stroke_rate',
        'stroke-rate': 'stroke_rate',
        'spm': 'stroke_rate',
        'heart rate': 'heart_rate',
        'heart_rate': 'heart_rate',
        'hr': 'heart_rate',
        'bpm': 'heart_rate',
        'distance (meters)': 'distance',
        'distance (m)': 'distance',
        'distance': 'distance'
    }, inplace=True)
    
    # Validate required columns
    if 'watts' not in df.columns:
        raise ValueError(f"File {path} missing required 'watts' column. Found: {list(df.columns)}")
    
    # Convert numeric columns
    numeric_cols = ['watts', 'pace', 'stroke_rate', 'heart_rate', 'time', 'distance']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows where watts is NaN
    initial_len = len(df)
    df = df.dropna(subset=['watts'])
    dropped = initial_len - len(df)
    
    if dropped > 0:
        print(f"   Dropped {dropped} rows with missing watts values")
    
    return df


# =====================================================================
# FEATURE EXTRACTION
# =====================================================================

def extract_window_features(window_df: pd.DataFrame) -> dict:
    """
    Extract features from a single window of data.
    
    MUST MATCH infer_models.py feature extraction!
    
    Args:
        window_df: DataFrame slice for one window
        
    Returns:
        Dictionary of features
    """
    watts = window_df["watts"].values
    
    features = {
        # Basic statistics
        "watts_mean": float(np.nanmean(watts)),
        "watts_std": float(np.nanstd(watts)),
        "watts_min": float(np.nanmin(watts)),
        "watts_max": float(np.nanmax(watts)),
        
        # Pace (handle missing)
        "pace_mean": float(np.nanmean(window_df["pace"].values)) if "pace" in window_df.columns else np.nan,
        "pace_std": float(np.nanstd(window_df["pace"].values)) if "pace" in window_df.columns else np.nan,
        
        # Stroke rate (handle missing)
        "sr_mean": float(np.nanmean(window_df["stroke_rate"].values)) if "stroke_rate" in window_df.columns else np.nan,
        "sr_std": float(np.nanstd(window_df["stroke_rate"].values)) if "stroke_rate" in window_df.columns else np.nan,
        
        # Heart rate (handle missing)
        "hr_mean": float(np.nanmean(window_df["heart_rate"].values)) if "heart_rate" in window_df.columns else np.nan,
        "hr_std": float(np.nanstd(window_df["heart_rate"].values)) if "heart_rate" in window_df.columns else np.nan,
        
        # Trend: linear slope of watts over time
        "watts_slope": float(np.polyfit(np.arange(len(watts)), watts, 1)[0]) if len(watts) > 1 else 0.0,
        
        # Jitter: mean absolute change in watts (smoothness)
        "watts_jitter": float(np.mean(np.abs(np.diff(watts)))) if len(watts) > 1 else 0.0,
        
        # Coefficient of variation (normalized variability)
        "watts_cv": float(np.nanstd(watts) / np.nanmean(watts)) if np.nanmean(watts) > 0 else 0.0,
    }
    
    # Replace NaN/inf with 0 (models expect numeric)
    for key, value in features.items():
        if not np.isfinite(value):
            features[key] = 0.0
    
    return features


def window_features(
    df: pd.DataFrame, 
    window: int = WINDOW_SIZE, 
    step: int = STEP_SIZE
) -> Tuple[pd.DataFrame, List[Tuple[int, int]]]:
    """
    Extract features using sliding window approach.
    
    Args:
        df: Input DataFrame with rowing data
        window: Window size in rows
        step: Step size for sliding window
        
    Returns:
        Tuple of (features_df, window_indices)
    """
    if len(df) < window:
        print(f"   ⚠️  File has only {len(df)} rows (need {window}). Skipping.")
        return pd.DataFrame(), []
    
    features = []
    idxs = []
    
    for start in range(0, len(df) - window + 1, step):
        w = df.iloc[start:start + window]
        
        # Extract features
        feat = extract_window_features(w)
        feat["start"] = start
        feat["end"] = start + window - 1
        
        features.append(feat)
        idxs.append((start, start + window - 1))
    
    feat_df = pd.DataFrame(features)
    
    return feat_df, idxs


# =====================================================================
# LABEL GENERATION
# =====================================================================

def generate_interval_labels(df: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate work/rest interval labels (binary classification).
    
    Label = 1 if window has above-median watts (work)
    Label = 0 if window has below-median watts (rest)
    
    Args:
        df: Original data
        feat_df: Features DataFrame
        
    Returns:
        Features with 'is_work' label added
    """
    threshold = df["watts"].quantile(WORK_THRESHOLD_PERCENTILE / 100.0)
    feat_df["is_work"] = (feat_df["watts_mean"] > threshold).astype(int)
    
    work_count = feat_df["is_work"].sum()
    rest_count = len(feat_df) - work_count
    
    print(f"   Labels: {work_count} work windows, {rest_count} rest windows (threshold: {threshold:.1f}W)")
    
    return feat_df


def generate_stroke_quality_target(df: pd.DataFrame) -> float:
    """
    Generate stroke quality target score (regression target).
    
    Rule-based scoring combining:
    - Efficiency (watts per stroke rate)
    - Consistency (low variability)
    - Smoothness (low jitter)
    
    Returns:
        Score from 0-100
    """
    watts = df["watts"].values
    
    # 1. Consistency: lower rolling std = better
    rolling_std = pd.Series(watts).rolling(10, min_periods=1).std().dropna().mean()
    
    # 2. Smoothness: lower jitter = better
    watts_jitter = pd.Series(watts).diff().abs().mean()
    
    # 3. Efficiency: watts per stroke rate
    if "stroke_rate" in df.columns:
        sr = df["stroke_rate"].mean()
        if sr and sr > 0:
            efficiency = watts.mean() / sr
        else:
            efficiency = watts.mean() / 150.0  # Fallback divisor
    else:
        efficiency = watts.mean() / 150.0
    
    # Normalize to 0-1 scale
    norm_eff = np.clip(
        (efficiency - EFFICIENCY_RANGE[0]) / (EFFICIENCY_RANGE[1] - EFFICIENCY_RANGE[0]), 
        0, 1
    )
    
    norm_cons = np.clip(
        1 - (rolling_std - CONSISTENCY_RANGE[0]) / (CONSISTENCY_RANGE[1] - CONSISTENCY_RANGE[0]), 
        0, 1
    )
    
    norm_smooth = np.clip(
        1 - (watts_jitter - SMOOTHNESS_RANGE[0]) / (SMOOTHNESS_RANGE[1] - SMOOTHNESS_RANGE[0]), 
        0, 1
    )
    
    # Weighted combination
    overall = (0.4 * norm_eff + 0.35 * norm_cons + 0.25 * norm_smooth) * 100
    
    return round(float(overall), 2)


def add_stroke_quality_targets(df: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add stroke quality target to each window.
    
    Computes a single score for the entire workout and assigns it to all windows.
    (Alternative: could compute per-window scores for more granular training)
    
    Args:
        df: Original data
        feat_df: Features DataFrame
        
    Returns:
        Features with 'stroke_quality_target' column added
    """
    sq_target = generate_stroke_quality_target(df)
    feat_df["stroke_quality_target"] = sq_target
    
    print(f"   Stroke quality target: {sq_target:.1f}/100")
    
    return feat_df


# =====================================================================
# PROCESSING PIPELINE
# =====================================================================

def process_single_file(path: str) -> Optional[pd.DataFrame]:
    """
    Process a single CSV file to extract features and labels.
    
    Args:
        path: Path to CSV file
        
    Returns:
        DataFrame with features and labels, or None if processing failed
    """
    filename = os.path.basename(path)
    print(f"\n📄 Processing: {filename}")
    
    try:
        # Load and normalize
        df = load_and_normalize(path)
        print(f"   Loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Check minimum length
        if len(df) < MIN_FILE_LENGTH:
            print(f"   ⚠️  File too short ({len(df)} rows, need {MIN_FILE_LENGTH}). Skipping.")
            return None
        
        # Extract windowed features
        feat_df, idxs = window_features(df, window=WINDOW_SIZE, step=STEP_SIZE)
        
        if feat_df.empty:
            print(f"   ⚠️  No windows extracted. Skipping.")
            return None
        
        print(f"   Extracted {len(feat_df)} windows (window={WINDOW_SIZE}, step={STEP_SIZE})")
        
        # Generate labels
        feat_df = generate_interval_labels(df, feat_df)
        feat_df = add_stroke_quality_targets(df, feat_df)
        
        # Add metadata
        feat_df["source_file"] = filename
        feat_df["original_length"] = len(df)
        
        print(f"   ✅ Success: {len(feat_df)} training samples")
        
        return feat_df
    
    except Exception as e:
        print(f"   ❌ Error processing {filename}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_all_files(input_paths: List[str]) -> pd.DataFrame:
    """
    Process all CSV files and combine into single dataset.
    
    Args:
        input_paths: List of paths to CSV files
        
    Returns:
        Combined DataFrame with all features and labels
    """
    all_rows = []
    successful = 0
    failed = 0
    
    print("=" * 70)
    print("🚣 ROWING DATA FEATURE EXTRACTION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Input files: {len(input_paths)}")
    print(f"  Window size: {WINDOW_SIZE} rows")
    print(f"  Step size: {STEP_SIZE} rows")
    print(f"  Min file length: {MIN_FILE_LENGTH} rows")
    print(f"  Output directory: {OUT_DIR}")
    
    for path in input_paths:
        result = process_single_file(path)
        
        if result is not None:
            all_rows.append(result)
            successful += 1
        else:
            failed += 1
    
    # Combine all results
    if not all_rows:
        print("\n" + "=" * 70)
        print("❌ ERROR: No usable data found!")
        print("=" * 70)
        print("\nTroubleshooting:")
        print("  1. Check that sample_data/*.csv files exist")
        print("  2. Ensure CSV files have 'watts' column")
        print(f"  3. Files must have at least {MIN_FILE_LENGTH} rows")
        print("=" * 70)
        return pd.DataFrame()
    
    combined = pd.concat(all_rows, ignore_index=True)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Files processed: {successful} successful, {failed} failed")
    print(f"Total windows: {len(combined)}")
    print(f"Features per window: {len([c for c in combined.columns if c not in ['source_file', 'original_length', 'is_work', 'stroke_quality_target', 'start', 'end']])}")
    print(f"\nTarget distribution:")
    print(f"  Work windows: {combined['is_work'].sum()} ({combined['is_work'].mean()*100:.1f}%)")
    print(f"  Rest windows: {(1-combined['is_work']).sum()} ({(1-combined['is_work'].mean())*100:.1f}%)")
    print(f"  Stroke quality: mean={combined['stroke_quality_target'].mean():.1f}, std={combined['stroke_quality_target'].std():.1f}")
    
    return combined


# =====================================================================
# SAVING & EXPORT
# =====================================================================

def save_datasets(combined: pd.DataFrame):
    """
    Save processed data to multiple formats.
    
    Args:
        combined: Combined features and labels DataFrame
    """
    if combined.empty:
        return
    
    # Main windows file
    windows_path = OUT_DIR / "windows.csv"
    combined.to_csv(windows_path, index=False)
    print(f"\n✅ Saved: {windows_path}")
    print(f"   Shape: {combined.shape}")
    
    # Separate features and labels for easier model training
    feature_cols = [c for c in combined.columns if c not in [
        'is_work', 'stroke_quality_target', 'source_file', 'original_length', 'start', 'end'
    ]]
    
    # Features only
    features_path = OUT_DIR / "features.csv"
    combined[feature_cols].to_csv(features_path, index=False)
    print(f"✅ Saved: {features_path} ({len(feature_cols)} features)")
    
    # Interval labels
    interval_labels_path = OUT_DIR / "interval_labels.csv"
    combined[['is_work']].to_csv(interval_labels_path, index=False)
    print(f"✅ Saved: {interval_labels_path}")
    
    # Stroke quality labels
    sq_labels_path = OUT_DIR / "stroke_quality_labels.csv"
    combined[['stroke_quality_target']].to_csv(sq_labels_path, index=False)
    print(f"✅ Saved: {sq_labels_path}")
    
    # Summary statistics
    summary_path = OUT_DIR / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("FEATURE EXTRACTION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples: {len(combined)}\n")
        f.write(f"Features: {len(feature_cols)}\n")
        f.write(f"Files processed: {combined['source_file'].nunique()}\n\n")
        f.write("Feature columns:\n")
        for col in feature_cols:
            f.write(f"  - {col}\n")
        f.write(f"\nTarget: is_work\n")
        f.write(f"  Work: {combined['is_work'].sum()} ({combined['is_work'].mean()*100:.1f}%)\n")
        f.write(f"  Rest: {(1-combined['is_work']).sum()} ({(1-combined['is_work'].mean())*100:.1f}%)\n")
        f.write(f"\nTarget: stroke_quality_target\n")
        f.write(f"  Mean: {combined['stroke_quality_target'].mean():.2f}\n")
        f.write(f"  Std: {combined['stroke_quality_target'].std():.2f}\n")
        f.write(f"  Min: {combined['stroke_quality_target'].min():.2f}\n")
        f.write(f"  Max: {combined['stroke_quality_target'].max():.2f}\n")
    
    print(f"✅ Saved: {summary_path}")
    
    # Feature correlation matrix (optional, for analysis)
    try:
        corr_path = OUT_DIR / "feature_correlations.csv"
        corr_matrix = combined[feature_cols].corr()
        corr_matrix.to_csv(corr_path)
        print(f"✅ Saved: {corr_path}")
    except:
        pass


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    """Main execution function"""
    
    # Check if input files exist
    if not INPUT_PATHS:
        print("❌ ERROR: No CSV files found in sample_data/")
        print("\nPlease add rowing workout CSV files to the sample_data/ directory.")
        return
    
    # Process all files
    combined = process_all_files(INPUT_PATHS)
    
    # Save results
    if not combined.empty:
        save_datasets(combined)
        
        print("\n" + "=" * 70)
        print("✅ FEATURE EXTRACTION COMPLETE")
        print("=" * 70)
        print(f"\nNext steps:")
        print(f"  1. Review {OUT_DIR}/windows.csv")
        print(f"  2. Train models using the extracted features")
        print(f"  3. Save trained models to models/ directory")
        print(f"  4. Run inference with infer_models.py")
        print("=" * 70)


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()