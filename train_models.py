# train_models.py
"""
Train ML models for rowing ergometer data analysis.

Models:
1. interval_clf - Classifies work/rest intervals (Random Forest)
2. stroke_quality_reg - Predicts stroke quality scores (Random Forest Regressor)
3. form_iso - Detects form anomalies (Isolation Forest)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURATION
# =====================================================================

# Paths
DATA_DIR = Path("data_processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Input files
WINDOWS_FILE = DATA_DIR / "windows.csv"

# Model output files
INTERVAL_CLF_PATH = MODELS_DIR / "interval_clf.joblib"
SQ_REG_PATH = MODELS_DIR / "stroke_quality_reg.joblib"
FORM_ISO_PATH = MODELS_DIR / "form_iso.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"  # Optional: for feature scaling

# Feature sets (must match inference)
INTERVAL_FEATURES = ["watts_mean", "watts_std", "watts_slope", "watts_jitter", "sr_mean", "pace_mean"]
SQ_FEATURES = ["watts_mean", "watts_std", "watts_slope", "watts_jitter", "sr_mean", "pace_mean"]
ISO_FEATURES = ["watts_mean", "watts_std", "watts_slope", "watts_jitter", "sr_mean"]

# Training configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42
CROSS_VAL_FOLDS = 5
USE_GRID_SEARCH = False  # Set to True for hyperparameter tuning (slower)

# Model hyperparameters
INTERVAL_CLF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

SQ_REG_PARAMS = {
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

ISO_PARAMS = {
    'contamination': 0.1,  # Expected proportion of anomalies (10%)
    'max_samples': 256,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}


# =====================================================================
# DATA LOADING
# =====================================================================

def load_training_data():
    """
    Load processed features and labels.
    
    Returns:
        DataFrame with all features and labels
    """
    print("=" * 70)
    print("📊 LOADING TRAINING DATA")
    print("=" * 70)
    
    if not WINDOWS_FILE.exists():
        raise FileNotFoundError(
            f"Training data not found: {WINDOWS_FILE}\n"
            f"Please run: python features_and_labels.py"
        )
    
    df = pd.read_csv(WINDOWS_FILE)
    
    print(f"\n✅ Loaded: {WINDOWS_FILE}")
    print(f"   Shape: {df.shape}")
    print(f"   Samples: {len(df)}")
    print(f"   Files: {df['source_file'].nunique() if 'source_file' in df.columns else 'N/A'}")
    
    # Check for required columns
    required = ['is_work', 'stroke_quality_target'] + INTERVAL_FEATURES
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    print(f"\n📋 Target distributions:")
    print(f"   Work/Rest: {df['is_work'].value_counts().to_dict()}")
    print(f"   Stroke Quality: mean={df['stroke_quality_target'].mean():.1f}, std={df['stroke_quality_target'].std():.1f}")
    
    return df


def prepare_features(df, feature_cols):
    """
    Extract and clean features.
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        
    Returns:
        Feature matrix (X)
    """
    X = df[feature_cols].copy()
    
    # Fill NaN with 0 (matches inference behavior)
    X = X.fillna(0)
    
    # Check for infinite values
    if np.isinf(X.values).any():
        print("   ⚠️  Replacing infinite values with 0")
        X = X.replace([np.inf, -np.inf], 0)
    
    return X


# =====================================================================
# MODEL 1: INTERVAL CLASSIFIER
# =====================================================================

def train_interval_classifier(df):
    """
    Train Random Forest classifier for work/rest interval detection.
    
    Args:
        df: Training data
        
    Returns:
        Trained model
    """
    print("\n" + "=" * 70)
    print("🎯 TRAINING INTERVAL CLASSIFIER")
    print("=" * 70)
    
    # Prepare data
    X = prepare_features(df, INTERVAL_FEATURES)
    y = df['is_work'].values
    
    print(f"\nFeatures: {INTERVAL_FEATURES}")
    print(f"Samples: {len(X)}")
    print(f"Class balance: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Train model
    if USE_GRID_SEARCH:
        print("\n🔍 Running grid search (this may take a while)...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [5, 10, 20]
        }
        model = GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        print(f"Best params: {model.best_params_}")
        model = model.best_estimator_
    else:
        print("\n🎲 Training Random Forest Classifier...")
        model = RandomForestClassifier(**INTERVAL_CLF_PARAMS)
        model.fit(X_train, y_train)
    
    # Evaluate
    print("\n📈 Evaluation:")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=CROSS_VAL_FOLDS, scoring='accuracy')
    print(f"   Cross-val accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # Test set performance
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"   Test accuracy: {test_acc:.3f}")
    
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Rest', 'Work'], digits=3))
    
    print("   Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   [[TN={cm[0,0]:4d}  FP={cm[0,1]:4d}]")
    print(f"    [FN={cm[1,0]:4d}  TP={cm[1,1]:4d}]]")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': INTERVAL_FEATURES,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n   Feature Importance:")
    for _, row in feature_importance.iterrows():
        print(f"      {row['feature']:20s}: {row['importance']:.3f}")
    
    return model


# =====================================================================
# MODEL 2: STROKE QUALITY REGRESSOR
# =====================================================================

def train_stroke_quality_regressor(df):
    """
    Train Random Forest regressor for stroke quality prediction.
    
    Args:
        df: Training data
        
    Returns:
        Trained model
    """
    print("\n" + "=" * 70)
    print("📊 TRAINING STROKE QUALITY REGRESSOR")
    print("=" * 70)
    
    # Prepare data
    X = prepare_features(df, SQ_FEATURES)
    y = df['stroke_quality_target'].values
    
    print(f"\nFeatures: {SQ_FEATURES}")
    print(f"Samples: {len(X)}")
    print(f"Target range: [{y.min():.1f}, {y.max():.1f}]")
    print(f"Target mean: {y.mean():.1f}, std: {y.std():.1f}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Train model
    if USE_GRID_SEARCH:
        print("\n🔍 Running grid search (this may take a while)...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5, 10]
        }
        model = GridSearchCV(
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            param_grid,
            cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        print(f"Best params: {model.best_params_}")
        model = model.best_estimator_
    else:
        print("\n🎲 Training Random Forest Regressor...")
        model = RandomForestRegressor(**SQ_REG_PARAMS)
        model.fit(X_train, y_train)
    
    # Evaluate
    print("\n📈 Evaluation:")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=CROSS_VAL_FOLDS, scoring='neg_mean_absolute_error')
    print(f"   Cross-val MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")
    
    # Test set performance
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"   Test MAE:  {mae:.2f}")
    print(f"   Test RMSE: {rmse:.2f}")
    print(f"   Test R²:   {r2:.3f}")
    
    # Residual analysis
    residuals = y_test - y_pred
    print(f"\n   Residuals:")
    print(f"      Mean: {residuals.mean():.2f}")
    print(f"      Std:  {residuals.std():.2f}")
    print(f"      Min:  {residuals.min():.2f}")
    print(f"      Max:  {residuals.max():.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': SQ_FEATURES,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n   Feature Importance:")
    for _, row in feature_importance.iterrows():
        print(f"      {row['feature']:20s}: {row['importance']:.3f}")
    
    return model


# =====================================================================
# MODEL 3: FORM ANOMALY DETECTOR (ISOLATION FOREST)
# =====================================================================

def train_form_anomaly_detector(df):
    """
    Train Isolation Forest for anomaly detection.
    
    Args:
        df: Training data
        
    Returns:
        Trained model
    """
    print("\n" + "=" * 70)
    print("🔍 TRAINING FORM ANOMALY DETECTOR")
    print("=" * 70)
    
    # Prepare data (use only "normal" data - work intervals)
    # Isolation Forest learns from normal data and flags outliers
    normal_data = df[df['is_work'] == 1].copy()  # Use work intervals as "normal"
    
    X = prepare_features(normal_data, ISO_FEATURES)
    
    print(f"\nFeatures: {ISO_FEATURES}")
    print(f"Training samples (work intervals): {len(X)}")
    print(f"Expected anomaly rate: {ISO_PARAMS['contamination']*100:.0f}%")
    
    # Train model
    print("\n🎲 Training Isolation Forest...")
    model = IsolationForest(**ISO_PARAMS)
    model.fit(X)
    
    # Evaluate on full dataset
    print("\n📈 Evaluation:")
    
    X_full = prepare_features(df, ISO_FEATURES)
    predictions = model.predict(X_full)
    
    n_anomalies = (predictions == -1).sum()
    n_normal = (predictions == 1).sum()
    
    print(f"   Predictions on full dataset:")
    print(f"      Normal:    {n_normal:5d} ({n_normal/len(predictions)*100:.1f}%)")
    print(f"      Anomalies: {n_anomalies:5d} ({n_anomalies/len(predictions)*100:.1f}%)")
    
    # Check anomalies in work vs rest
    if 'is_work' in df.columns:
        work_mask = df['is_work'] == 1
        work_anomalies = (predictions[work_mask] == -1).sum()
        rest_anomalies = (predictions[~work_mask] == -1).sum()
        
        print(f"\n   Anomalies by interval type:")
        print(f"      Work intervals: {work_anomalies}/{work_mask.sum()} ({work_anomalies/work_mask.sum()*100:.1f}%)")
        print(f"      Rest intervals: {rest_anomalies}/{(~work_mask).sum()} ({rest_anomalies/(~work_mask).sum()*100:.1f}%)")
    
    # Anomaly scores
    scores = model.score_samples(X_full)
    print(f"\n   Anomaly scores:")
    print(f"      Mean:   {scores.mean():.3f}")
    print(f"      Std:    {scores.std():.3f}")
    print(f"      Min:    {scores.min():.3f} (most anomalous)")
    print(f"      Max:    {scores.max():.3f} (most normal)")
    
    return model


# =====================================================================
# MODEL SAVING
# =====================================================================

def save_models(interval_clf, sq_reg, form_iso):
    """
    Save all trained models to disk.
    
    Args:
        interval_clf: Trained interval classifier
        sq_reg: Trained stroke quality regressor
        form_iso: Trained anomaly detector
    """
    print("\n" + "=" * 70)
    print("💾 SAVING MODELS")
    print("=" * 70)
    
    models_saved = []
    
    # Save interval classifier
    if interval_clf is not None:
        joblib.dump(interval_clf, INTERVAL_CLF_PATH)
        print(f"✅ Saved: {INTERVAL_CLF_PATH}")
        models_saved.append(INTERVAL_CLF_PATH.name)
    
    # Save stroke quality regressor
    if sq_reg is not None:
        joblib.dump(sq_reg, SQ_REG_PATH)
        print(f"✅ Saved: {SQ_REG_PATH}")
        models_saved.append(SQ_REG_PATH.name)
    
    # Save anomaly detector
    if form_iso is not None:
        joblib.dump(form_iso, FORM_ISO_PATH)
        print(f"✅ Saved: {FORM_ISO_PATH}")
        models_saved.append(FORM_ISO_PATH.name)
    
    print(f"\n✅ Total models saved: {len(models_saved)}")
    
    return models_saved


# =====================================================================
# MODEL INFO
# =====================================================================

def generate_model_info(interval_clf, sq_reg, form_iso, df):
    """
    Generate a summary file with model information.
    
    Args:
        interval_clf: Trained interval classifier
        sq_reg: Trained stroke quality regressor
        form_iso: Trained anomaly detector
        df: Training data
    """
    info_path = MODELS_DIR / "model_info.txt"
    
    with open(info_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ROWING ML MODELS - SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Training date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training samples: {len(df)}\n")
        f.write(f"Source files: {df['source_file'].nunique() if 'source_file' in df.columns else 'N/A'}\n\n")
        
        # Model 1: Interval Classifier
        f.write("-" * 70 + "\n")
        f.write("MODEL 1: Interval Classifier (interval_clf.joblib)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Type: Random Forest Classifier\n")
        f.write(f"Task: Classify work/rest intervals\n")
        f.write(f"Features: {', '.join(INTERVAL_FEATURES)}\n")
        f.write(f"Output: 0=rest, 1=work\n")
        if interval_clf:
            f.write(f"Estimators: {interval_clf.n_estimators}\n")
            f.write(f"Max depth: {interval_clf.max_depth}\n")
        f.write("\n")
        
        # Model 2: Stroke Quality Regressor
        f.write("-" * 70 + "\n")
        f.write("MODEL 2: Stroke Quality Regressor (stroke_quality_reg.joblib)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Type: Random Forest Regressor\n")
        f.write(f"Task: Predict stroke quality score (0-100)\n")
        f.write(f"Features: {', '.join(SQ_FEATURES)}\n")
        f.write(f"Output: Float score 0-100\n")
        if sq_reg:
            f.write(f"Estimators: {sq_reg.n_estimators}\n")
            f.write(f"Max depth: {sq_reg.max_depth}\n")
        f.write("\n")
        
        # Model 3: Form Anomaly Detector
        f.write("-" * 70 + "\n")
        f.write("MODEL 3: Form Anomaly Detector (form_iso.joblib)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Type: Isolation Forest\n")
        f.write(f"Task: Detect anomalous form/technique\n")
        f.write(f"Features: {', '.join(ISO_FEATURES)}\n")
        f.write(f"Output: -1=anomaly, 1=normal\n")
        if form_iso:
            f.write(f"Contamination: {form_iso.contamination}\n")
            f.write(f"Max samples: {form_iso.max_samples}\n")
        f.write("\n")
        
        f.write("=" * 70 + "\n")
        f.write("USAGE\n")
        f.write("=" * 70 + "\n\n")
        f.write("Load models:\n")
        f.write("    from infer_models import load_models\n")
        f.write("    load_models()\n\n")
        f.write("Run inference:\n")
        f.write("    from infer_models import run_all_models\n")
        f.write("    results = run_all_models(df)\n\n")
        f.write("Integrate with backend:\n")
        f.write("    See app.py for integration example\n\n")
    
    print(f"✅ Saved: {info_path}")


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    """Main training pipeline"""
    
    print("\n")
    print("=" * 70)
    print("🚣 ROWING ML MODEL TRAINING PIPELINE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Data source: {WINDOWS_FILE}")
    print(f"  Models output: {MODELS_DIR}")
    print(f"  Test size: {TEST_SIZE*100:.0f}%")
    print(f"  Cross-validation folds: {CROSS_VAL_FOLDS}")
    print(f"  Random state: {RANDOM_STATE}")
    print(f"  Grid search: {'Enabled' if USE_GRID_SEARCH else 'Disabled'}")
    
    # Load data
    try:
        df = load_training_data()
    except Exception as e:
        print(f"\n❌ Failed to load data: {e}")
        return
    
    # Train models
    interval_clf = None
    sq_reg = None
    form_iso = None
    
    try:
        interval_clf = train_interval_classifier(df)
    except Exception as e:
        print(f"\n❌ Failed to train interval classifier: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        sq_reg = train_stroke_quality_regressor(df)
    except Exception as e:
        print(f"\n❌ Failed to train stroke quality regressor: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        form_iso = train_form_anomaly_detector(df)
    except Exception as e:
        print(f"\n❌ Failed to train anomaly detector: {e}")
        import traceback
        traceback.print_exc()
    
    # Save models
    if interval_clf or sq_reg or form_iso:
        saved = save_models(interval_clf, sq_reg, form_iso)
        generate_model_info(interval_clf, sq_reg, form_iso, df)
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE")
        print("=" * 70)
        print(f"\nModels saved to: {MODELS_DIR}/")
        for model_name in saved:
            print(f"  ✅ {model_name}")
        
        print("\n📝 Next steps:")
        print("  1. Test models: python infer_models.py")
        print("  2. Start backend: python app.py")
        print("  3. Upload workout CSV to test end-to-end")
        print("\n" + "=" * 70)
    else:
        print("\n❌ No models were successfully trained")


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()