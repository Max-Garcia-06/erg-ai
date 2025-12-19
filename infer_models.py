# infer_models.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# =====================================================================
# CONFIGURATION
# =====================================================================
MODELS_DIR = Path("models")

# Global model instances (loaded once at startup)
_interval_clf = None
_sq_reg = None
_form_iso = None

# Feature columns (must match training data)
INTERVAL_FEATURES = ["watts_mean", "watts_std", "watts_slope", "watts_jitter", "sr_mean", "pace_mean"]
SQ_FEATURES = ["watts_mean", "watts_std", "watts_slope", "watts_jitter", "sr_mean", "pace_mean"]
ISO_FEATURES = ["watts_mean", "watts_std", "watts_slope", "watts_jitter", "sr_mean"]

# Window configuration (must match training)
DEFAULT_WINDOW = 30
DEFAULT_STEP = 5


# =====================================================================
# MODEL LOADING
# =====================================================================
def load_models():
    """
    Load all trained models from disk.
    Called once at application startup.
    Models are stored in global variables for reuse.
    """
    global _interval_clf, _sq_reg, _form_iso
    
    models_loaded = []
    
    # Load interval classifier
    if _interval_clf is None:
        interval_path = MODELS_DIR / "interval_clf.joblib"
        if interval_path.exists():
            try:
                _interval_clf = joblib.load(interval_path)
                models_loaded.append("interval_clf")
                print(f"✅ Loaded: {interval_path}")
            except Exception as e:
                print(f"⚠️ Failed to load interval_clf: {e}")
        else:
            print(f"⚠️ Model not found: {interval_path}")
    
    # Load stroke quality regressor
    if _sq_reg is None:
        sq_path = MODELS_DIR / "stroke_quality_reg.joblib"
        if sq_path.exists():
            try:
                _sq_reg = joblib.load(sq_path)
                models_loaded.append("stroke_quality_reg")
                print(f"✅ Loaded: {sq_path}")
            except Exception as e:
                print(f"⚠️ Failed to load stroke_quality_reg: {e}")
        else:
            print(f"⚠️ Model not found: {sq_path}")
    
    # Load form anomaly detector (Isolation Forest)
    if _form_iso is None:
        iso_path = MODELS_DIR / "form_iso.joblib"
        if iso_path.exists():
            try:
                _form_iso = joblib.load(iso_path)
                models_loaded.append("form_iso")
                print(f"✅ Loaded: {iso_path}")
            except Exception as e:
                print(f"⚠️ Failed to load form_iso: {e}")
        else:
            print(f"⚠️ Model not found: {iso_path}")
    
    if not models_loaded:
        print("⚠️ No ML models loaded. Will use rule-based fallbacks.")
    else:
        print(f"🎯 ML models ready: {', '.join(models_loaded)}")
    
    return len(models_loaded) > 0


def models_available() -> Dict[str, bool]:
    """Check which models are currently loaded"""
    return {
        "interval_clf": _interval_clf is not None,
        "stroke_quality_reg": _sq_reg is not None,
        "form_iso": _form_iso is not None
    }


# =====================================================================
# FEATURE EXTRACTION
# =====================================================================
def model_features_from_df(
    df: pd.DataFrame, 
    window: int = DEFAULT_WINDOW, 
    step: int = DEFAULT_STEP
) -> Tuple[pd.DataFrame, List[Tuple[int, int]]]:
    """
    Extract windowed features from a rowing workout DataFrame.
    
    Args:
        df: DataFrame with columns: watts, pace, stroke_rate, heart_rate, etc.
        window: Window size in rows (default: 30)
        step: Step size between windows (default: 5)
    
    Returns:
        Tuple of (features_df, window_indices)
        - features_df: DataFrame with one row per window
        - window_indices: List of (start, end) tuples for each window
    """
    
    # Validate input
    if "watts" not in df.columns:
        raise ValueError("DataFrame must contain 'watts' column")
    
    if len(df) < window:
        raise ValueError(f"DataFrame has {len(df)} rows, need at least {window}")
    
    X_windows = []
    idxs = []
    
    # Slide window across data
    for start in range(0, len(df) - window + 1, step):
        w = df.iloc[start:start + window]
        
        # Extract features from this window
        feat = extract_window_features(w)
        feat["start"] = start
        feat["end"] = start + window - 1
        
        X_windows.append(feat)
        idxs.append((start, start + window - 1))
    
    # Convert to DataFrame and fill missing values
    feat_df = pd.DataFrame(X_windows).fillna(0)
    
    return feat_df, idxs


def extract_window_features(window_df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract features from a single window of data.
    
    This must match the feature extraction in features_and_labels.py!
    """
    watts = window_df["watts"].values
    
    features = {
        # Basic statistics
        "watts_mean": float(np.nanmean(watts)),
        "watts_std": float(np.nanstd(watts)),
        
        # Pace (handle missing)
        "pace_mean": float(np.nanmean(window_df["pace"].values)) if "pace" in window_df.columns else 0.0,
        
        # Stroke rate (handle missing)
        "sr_mean": float(np.nanmean(window_df["stroke_rate"].values)) if "stroke_rate" in window_df.columns else 0.0,
        
        # Heart rate (handle missing) - ✅ NOW INCLUDED
        "hr_mean": float(np.nanmean(window_df["heart_rate"].values)) if "heart_rate" in window_df.columns else 0.0,
        
        # Trend: linear slope of watts over time
        "watts_slope": float(np.polyfit(np.arange(len(watts)), watts, 1)[0]) if len(watts) > 1 else 0.0,
        
        # Jitter: mean absolute change in watts
        "watts_jitter": float(np.mean(np.abs(np.diff(watts)))) if len(watts) > 1 else 0.0,
    }
    
    # Replace NaN/inf with 0
    for key, value in features.items():
        if not np.isfinite(value):
            features[key] = 0.0
    
    return features


# =====================================================================
# PREDICTION FUNCTIONS
# =====================================================================
def predict_intervals(feat_df: pd.DataFrame, idxs: List[Tuple[int, int]]) -> List[List[int]]:
    """
    Predict work/rest intervals using the interval classifier.
    
    Returns:
        List of [start, end] intervals in original DataFrame row indices
    """
    if _interval_clf is None:
        print("⚠️ interval_clf not loaded, returning empty intervals")
        return []
    
    try:
        # Get predictions (1 = work, 0 = rest)
        X = feat_df[INTERVAL_FEATURES].fillna(0)
        preds = _interval_clf.predict(X)
        
        # Find all work windows
        work_idx = [i for i, p in enumerate(preds) if p == 1]
        
        if not work_idx:
            return []
        
        # Group contiguous work windows into intervals
        groups = []
        cur = [work_idx[0]]
        
        for i in work_idx[1:]:
            if i == cur[-1] + 1:
                # Contiguous - add to current group
                cur.append(i)
            else:
                # Gap - start new group
                groups.append((cur[0], cur[-1]))
                cur = [i]
        
        # Don't forget the last group
        groups.append((cur[0], cur[-1]))
        
        # Convert window indices to row indices
        intervals = []
        for g_start, g_end in groups:
            row_start = idxs[g_start][0]
            row_end = idxs[g_end][1]
            intervals.append([int(row_start), int(row_end)])
        
        return intervals
    
    except Exception as e:
        print(f"⚠️ Interval prediction failed: {e}")
        return []


def predict_stroke_quality(feat_df: pd.DataFrame) -> Optional[float]:
    """
    Predict overall stroke quality score using regression model.
    
    Returns:
        Float score (0-100) or None if prediction fails
    """
    if _sq_reg is None:
        print("⚠️ stroke_quality_reg not loaded")
        return None
    
    try:
        # Get predictions for all windows
        X = feat_df[SQ_FEATURES].fillna(0)
        sq_preds = _sq_reg.predict(X)
        
        # Average across all windows
        sq_mean = float(np.mean(sq_preds))
        
        # Clamp to 0-100 range
        sq_mean = max(0.0, min(100.0, sq_mean))
        
        return round(sq_mean, 1)
    
    except Exception as e:
        print(f"⚠️ Stroke quality prediction failed: {e}")
        return None


def predict_anomalies(feat_df: pd.DataFrame, idxs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Detect anomalous windows using Isolation Forest.
    
    Returns:
        List of (start, end) tuples for anomalous windows
    """
    if _form_iso is None:
        print("⚠️ form_iso not loaded")
        return []
    
    try:
        # Get predictions (-1 = anomaly, 1 = normal)
        X = feat_df[ISO_FEATURES].fillna(0)
        iso_preds = _form_iso.predict(X)
        
        # Find anomalous windows
        anomaly_windows = [idxs[i] for i, pred in enumerate(iso_preds) if pred == -1]
        
        return anomaly_windows
    
    except Exception as e:
        print(f"⚠️ Anomaly prediction failed: {e}")
        return []


# =====================================================================
# MAIN INFERENCE FUNCTION
# =====================================================================
def run_all_models(df: pd.DataFrame, window: int = DEFAULT_WINDOW, step: int = DEFAULT_STEP) -> Dict:
    """
    Run all available ML models on a workout DataFrame.
    
    Args:
        df: Rowing workout data (must have 'watts' column)
        window: Window size for feature extraction
        step: Step size for sliding window
    
    Returns:
        Dictionary with predictions:
        {
            "intervals_ml": [[start, end], ...],
            "stroke_quality_pred": float or None,
            "form_anomalies": [(start, end), ...],
            "windows_analyzed": int
        }
    """
    
    # Initialize results
    results = {
        "intervals_ml": [],
        "stroke_quality_pred": None,
        "form_anomalies": [],
        "windows_analyzed": 0
    }
    
    try:
        # Extract features
        feat_df, idxs = model_features_from_df(df, window=window, step=step)
        results["windows_analyzed"] = len(feat_df)
        
        if feat_df.empty:
            print("⚠️ No windows extracted")
            return results
        
        # Run each model
        if _interval_clf is not None:
            results["intervals_ml"] = predict_intervals(feat_df, idxs)
        
        if _sq_reg is not None:
            results["stroke_quality_pred"] = predict_stroke_quality(feat_df)
        
        if _form_iso is not None:
            results["form_anomalies"] = predict_anomalies(feat_df, idxs)
        
        return results
    
    except Exception as e:
        print(f"❌ run_all_models failed: {e}")
        return results


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================
def get_model_info() -> Dict:
    """Get information about loaded models"""
    available = models_available()
    
    info = {
        "models_loaded": sum(available.values()),
        "models_available": available,
        "window_size": DEFAULT_WINDOW,
        "step_size": DEFAULT_STEP,
        "feature_sets": {
            "interval_features": INTERVAL_FEATURES,
            "stroke_quality_features": SQ_FEATURES,
            "anomaly_features": ISO_FEATURES
        }
    }
    
    return info


# =====================================================================
# TESTING (if run directly)
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing infer_models.py")
    print("=" * 60)
    
    # Load models
    print("\n1. Loading models...")
    load_models()
    
    # Show status
    print("\n2. Model status:")
    info = get_model_info()
    print(f"   Models loaded: {info['models_loaded']}/3")
    for model_name, loaded in info['models_available'].items():
        status = "✅" if loaded else "❌"
        print(f"   {status} {model_name}")
    
    # Try loading sample data
    print("\n3. Testing with sample data...")
    sample_files = list(Path("sample_data").glob("*.csv"))
    
    if sample_files:
        test_file = sample_files[0]
        print(f"   Loading: {test_file}")
        
        try:
            # Load and normalize
            df = pd.read_csv(test_file)
            df.columns = df.columns.str.strip().str.lower()
            df.rename(columns={
                'time (s)': 'time',
                'pace (sec/500m)': 'pace',
                'stroke rate': 'stroke_rate',
                'heart rate': 'heart_rate',
            }, inplace=True)
            
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            
            # Run models
            print("\n4. Running inference...")
            results = run_all_models(df)
            
            print(f"\n5. Results:")
            print(f"   Windows analyzed: {results['windows_analyzed']}")
            print(f"   Intervals detected: {len(results['intervals_ml'])}")
            print(f"   Stroke quality: {results['stroke_quality_pred']}")
            print(f"   Anomalies found: {len(results['form_anomalies'])}")
            
            if results['intervals_ml']:
                print(f"\n   Intervals:")
                for i, (start, end) in enumerate(results['intervals_ml'], 1):
                    print(f"      {i}. Rows {start}-{end}")
            
            if results['form_anomalies']:
                print(f"\n   Anomalies:")
                for i, (start, end) in enumerate(results['form_anomalies'][:5], 1):
                    print(f"      {i}. Rows {start}-{end}")
                if len(results['form_anomalies']) > 5:
                    print(f"      ... and {len(results['form_anomalies']) - 5} more")
        
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("   ⚠️ No sample data found in sample_data/")
    
    print("\n" + "=" * 60)
    print("✅ Test complete")
    print("=" * 60)