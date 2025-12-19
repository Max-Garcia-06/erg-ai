# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import io
import math
import numpy as np
import os

# ✅ IMPORT YOUR ML INFERENCE MODULE
from infer_models import load_models, run_all_models

app = FastAPI(
    title="erg.ai", 
    description="Analyze rowing ergometer data with AI insights",
    version="2.0.0"
)

# ---------------- Config ----------------
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MIN_DATA_POINTS = 10
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# ---------------- CORS ----------------
allowed_origins = ["*"] if ENVIRONMENT == "development" else [
    "https://yourdomain.com",
    "https://www.yourdomain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Static Files ----------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "2.0.0", "ml_models_loaded": True}


# ---------------- Helper: Normalization ----------------
def normalize(value, min_val, max_val):
    """Normalize value to 0-1 range"""
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return 0.5
    except (ValueError, TypeError):
        return 0.5
    
    if max_val - min_val == 0:
        return 0.5
    
    return max(0.0, min(1.0, (v - min_val) / (max_val - min_val)))


# ---------------- RULE-BASED FALLBACKS (keep these) ----------------

def detect_intervals_rule_based(df):
    """Fallback: Rule-based interval detection"""
    if 'watts' not in df.columns:
        return {"intervals": [], "workout_type": "unknown"}
    
    watts = df["watts"].ffill().bfill().values
    
    if len(watts) == 0 or np.all(np.isnan(watts)):
        return {"intervals": [], "workout_type": "unknown"}
    
    intervals = []
    threshold = 60
    in_interval = False
    start = 0
    mean_watts = np.nanmean(watts)
    
    for i in range(1, len(watts)):
        if np.isnan(watts[i]) or np.isnan(watts[i-1]):
            continue
        
        diff = watts[i] - watts[i - 1]
        
        if not in_interval and watts[i] > mean_watts + 10:
            in_interval = True
            start = i
        elif in_interval and diff < -threshold:
            end = i - 1
            if end - start > 5:
                intervals.append([int(start), int(end)])
            in_interval = False
    
    if in_interval and len(watts) - start > 5:
        intervals.append([int(start), int(len(watts) - 1)])
    
    workout_type = "intervals" if len(intervals) > 1 else ("steady_state" if len(intervals) == 1 else "unknown")
    
    return {"intervals": intervals, "workout_type": workout_type}


def detect_anomalies_rule_based(df):
    """Fallback: Z-score anomaly detection"""
    if 'watts' not in df.columns:
        return []
    
    watts = df["watts"].ffill().bfill().values
    
    if len(watts) < 3:
        return []
    
    diffs = np.abs(np.diff(watts))
    diffs_clean = diffs[~np.isnan(diffs)]
    
    if len(diffs_clean) == 0:
        return []
    
    mean_diff = float(np.mean(diffs_clean))
    std_diff = float(np.std(diffs_clean))
    
    if std_diff == 0:
        return []
    
    threshold = mean_diff + 3 * std_diff
    anomalies = [int(i + 1) for i, d in enumerate(diffs) if not np.isnan(d) and d > threshold]
    
    return anomalies


# ---------------- Stroke Quality (Rule-based for comparison) ----------------
def compute_stroke_quality_rule_based(df):
    """Rule-based stroke quality (keep for comparison)"""
    if 'watts' not in df.columns:
        return {
            "overall": 0.0,
            "consistency": 0.0,
            "efficiency": 0.0,
            "drift": 0.0,
        }
    
    watts = df["watts"].ffill().bfill().values
    
    if len(watts) == 0 or np.all(np.isnan(watts)):
        return {
            "overall": 0.0,
            "consistency": 0.0,
            "efficiency": 0.0,
            "drift": 0.0,
        }
    
    sr_present = "stroke_rate" in df.columns and df["stroke_rate"].dropna().shape[0] > 0
    
    if sr_present:
        sr = df["stroke_rate"].ffill().bfill().values
        with np.errstate(divide="ignore", invalid="ignore"):
            eff_vals = np.where(sr > 0, watts / sr, np.nan)
        efficiency_raw = float(np.nanmean(eff_vals))
        efficiency_score = normalize(efficiency_raw, 5, 20)
    else:
        efficiency_raw = float(np.nanmean(watts))
        efficiency_score = normalize(efficiency_raw, 50, 400)
    
    rolling_std = pd.Series(watts).rolling(10, min_periods=1).std().fillna(0).values
    consistency_raw = float(np.nanmean(rolling_std))
    norm_consistency = 1.0 - normalize(consistency_raw, 5, 50)
    
    jerkiness = float(pd.Series(watts).diff().abs().mean())
    drift_score = 1.0 - normalize(jerkiness, 5, 50)
    
    overall = (0.40 * efficiency_score + 0.35 * norm_consistency + 0.25 * drift_score) * 100
    
    return {
        "overall": round(float(overall), 2),
        "consistency": round(float(norm_consistency * 100), 2),
        "efficiency": round(float(efficiency_score * 100), 2),
        "drift": round(float(drift_score * 100), 2),
    }


# ---------------- Fatigue Score ----------------
def compute_fatigue(df):
    """Calculate fatigue score"""
    if 'watts' not in df.columns:
        return 0.0
    
    watts = df["watts"].dropna().values
    
    if len(watts) < 2:
        return 0.0
    
    start_power = float(np.mean(watts[:min(10, len(watts))]))
    end_power = float(np.mean(watts[-min(10, len(watts)):]))
    drift = start_power - end_power
    norm_drift = normalize(drift, 0, 50)
    
    variability = float(np.std(watts))
    norm_variability = normalize(variability, 5, 40)
    
    if "heart_rate" in df.columns:
        hr_clean = df["heart_rate"].dropna()
        watts_clean = df.loc[hr_clean.index, "watts"].dropna()
        
        if len(hr_clean) > 10 and len(watts_clean) > 10:
            try:
                corr = float(hr_clean.corr(watts_clean))
                norm_decouple = normalize(1 - abs(corr), 0, 1)
            except:
                norm_decouple = 0.5
        else:
            norm_decouple = 0.5
    else:
        norm_decouple = 0.5
    
    fatigue_score = (0.4 * norm_drift + 0.3 * norm_variability + 0.3 * norm_decouple) * 100
    
    return round(float(fatigue_score), 1)


# ---------------- Main Analysis Engine ----------------
def analyze_erg_from_bytes(file_bytes):
    """Main analysis pipeline with ML integration"""
    
    # Parse CSV
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")
    
    # Validate data
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    
    if len(df) < MIN_DATA_POINTS:
        raise HTTPException(
            status_code=400, 
            detail=f"Not enough data points (found {len(df)}, need at least {MIN_DATA_POINTS})"
        )
    
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    df.rename(columns={
        'time (s)': 'time',
        'time (seconds)': 'time',
        'time(s)': 'time',
        'pace (sec/500m)': 'pace',
        'pace (seconds)': 'pace',
        'stroke rate': 'stroke_rate',
        'stroke-rate': 'stroke_rate',
        'heart rate': 'heart_rate',
    }, inplace=True)
    
    # Check required columns
    if "watts" not in df.columns:
        raise HTTPException(
            status_code=400, 
            detail=f"CSV missing required 'watts' column. Found: {', '.join(df.columns)}"
        )
    
    # Convert to numeric
    numeric_cols = ['watts', 'pace', 'heart_rate', 'stroke_rate', 'time']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # ========================================
    # ✅ RUN ML MODELS
    # ========================================
    ml_results = None
    use_ml = False
    
    try:
        ml_results = run_all_models(df)
        use_ml = True
        print("✅ ML models executed successfully")
    except Exception as e:
        print(f"⚠️ ML inference failed, using rule-based fallback: {e}")
        use_ml = False
    
    # ========================================
    # BASIC METRICS
    # ========================================
    avg_split = float(df["pace"].mean()) if "pace" in df.columns else None
    avg_power = float(df["watts"].mean())
    consistency = float(df["watts"].std())
    
    watts_clean = df["watts"].dropna()
    if len(watts_clean) >= 2:
        drift = float(watts_clean.iloc[-1] - watts_clean.iloc[0])
    else:
        drift = 0.0
    
    # ========================================
    # INTERVALS: ML or Rule-based
    # ========================================
    if use_ml and ml_results and ml_results.get("intervals_ml"):
        intervals = ml_results["intervals_ml"]
        workout_type = "intervals" if len(intervals) > 1 else "steady_state"
        interval_source = "ml"
    else:
        interval_data = detect_intervals_rule_based(df)
        intervals = interval_data["intervals"]
        workout_type = interval_data["workout_type"]
        interval_source = "rule_based"
    
    # ========================================
    # ANOMALIES: ML or Rule-based
    # ========================================
    if use_ml and ml_results and ml_results.get("form_anomalies"):
        # Convert window indices to row indices (use start of anomalous window)
        anomalies = [win[0] for win in ml_results["form_anomalies"]]
        anomaly_source = "ml"
    else:
        anomalies = detect_anomalies_rule_based(df)
        anomaly_source = "rule_based"
    
    # ========================================
    # STROKE QUALITY: ML Prediction (sq_pred)
    # ========================================
    sq_pred = None
    if use_ml and ml_results and ml_results.get("stroke_quality_pred") is not None:
        sq_pred = round(float(ml_results["stroke_quality_pred"]), 1)
    
    # Also compute rule-based for comparison
    stroke_quality_rules = compute_stroke_quality_rule_based(df)
    
    # ========================================
    # FATIGUE SCORE
    # ========================================
    fatigue_score = compute_fatigue(df)
    
    # ========================================
    # BUILD RESPONSE
    # ========================================
    return {
        "avg_split": round(avg_split, 2) if avg_split is not None and not np.isnan(avg_split) else None,
        "avg_power": round(avg_power, 1) if not np.isnan(avg_power) else None,
        "consistency": round(consistency, 2) if not np.isnan(consistency) else 0.0,
        "drift": round(drift, 2),
        "intervals": intervals,
        "workout_type": workout_type,
        "interval_report": f"{len(intervals)} interval(s) detected" if intervals else "No distinct intervals detected",
        "stroke_quality": stroke_quality_rules,  # Rule-based breakdown
        "sq_pred": sq_pred,  # ✅ ML prediction
        "anomalies": anomalies,
        "fatigue_score": fatigue_score,
        "data_points": len(df),
        "duration_seconds": float(df["time"].max()) if "time" in df.columns and not df["time"].isna().all() else None,
        # Metadata about which models were used
        "ml_used": use_ml,
        "interval_source": interval_source,
        "anomaly_source": anomaly_source
    }


# ---------------- API Endpoint ----------------
@app.post("/analyze")
async def analyze_workout(file: UploadFile = File(...)):
    """Analyze uploaded rowing workout CSV file"""
    
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")
    
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large (max {MAX_FILE_SIZE // 1024 // 1024}MB)"
        )
    
    try:
        result = analyze_erg_from_bytes(contents)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ---------------- Startup: Load Models ----------------
@app.on_event("startup")
async def startup_event():
    print("🚣 erg.ai starting up...")
    print(f"📁 Environment: {ENVIRONMENT}")
    print(f"🔒 CORS origins: {allowed_origins}")
    
    # ✅ LOAD ML MODELS
    try:
        load_models()
        print("✅ ML models loaded successfully")
    except Exception as e:
        print(f"⚠️ ML models failed to load (will use rule-based fallback): {e}")

@app.on_event("shutdown")
async def shutdown_event():
    print("🚣 erg.ai shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)