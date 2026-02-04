import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfigLoader:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        if not self.config_path.exists():
            logger.warning(
                f"Config file {self.config_path} not found. Using defaults."
            )
            return self._default_config()

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def _default_config() -> Dict:
        return {
            'models_dir': 'models',
            'window_size': 30,
            'step_size': 5,
            'interval_features': [
                'watts_mean', 'watts_std', 'watts_slope',
                'watts_jitter', 'sr_mean', 'pace_mean'
            ],
            'stroke_quality_features': [
                'watts_mean', 'watts_std', 'watts_slope',
                'watts_jitter', 'sr_mean', 'pace_mean'
            ],
            'anomaly_features': [
                'watts_mean', 'watts_std', 'watts_slope',
                'watts_jitter', 'sr_mean'
            ]
        }

    def get(self, key: str, default=None):
        return self.config.get(key, default)


class ModelLoader:
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)

    def load_model(self, model_name: str) -> Optional[object]:
        model_path = self.models_dir / f"{model_name}.joblib"

        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            return None

        try:
            model = joblib.load(model_path)
            logger.info(f"Loaded model: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            return None


class FeatureExtractor:
    @staticmethod
    def extract_window_features(window_df: pd.DataFrame) -> Dict:
        watts = window_df["watts"].values

        features = {
            "watts_mean": float(np.nanmean(watts)),
            "watts_std": float(np.nanstd(watts)),
            "watts_min": float(np.nanmin(watts)),
            "watts_max": float(np.nanmax(watts)),
            "pace_mean": (
                float(np.nanmean(window_df["pace"].values))
                if "pace" in window_df.columns else np.nan
            ),
            "pace_std": (
                float(np.nanstd(window_df["pace"].values))
                if "pace" in window_df.columns else np.nan
            ),
            "sr_mean": (
                float(np.nanmean(window_df["stroke_rate"].values))
                if "stroke_rate" in window_df.columns else np.nan
            ),
            "sr_std": (
                float(np.nanstd(window_df["stroke_rate"].values))
                if "stroke_rate" in window_df.columns else np.nan
            ),
            "hr_mean": (
                float(np.nanmean(window_df["heart_rate"].values))
                if "heart_rate" in window_df.columns else np.nan
            ),
            "hr_std": (
                float(np.nanstd(window_df["heart_rate"].values))
                if "heart_rate" in window_df.columns else np.nan
            ),
            "watts_slope": (
                float(np.polyfit(np.arange(len(watts)), watts, 1)[0])
                if len(watts) > 1 else 0.0
            ),
            "watts_jitter": (
                float(np.mean(np.abs(np.diff(watts))))
                if len(watts) > 1 else 0.0
            ),
            "watts_cv": (
                float(np.nanstd(watts) / np.nanmean(watts))
                if np.nanmean(watts) > 0 else 0.0
            ),
        }

        # Replace NaN/inf with 0
        for key, value in features.items():
            if not np.isfinite(value):
                features[key] = 0.0

        return features

    def extract_from_dataframe(
        self,
        df: pd.DataFrame,
        window_size: int,
        step_size: int
    ) -> Tuple[pd.DataFrame, List[Tuple[int, int]]]:
        if len(df) < window_size:
            logger.warning(
                f"DataFrame too short ({len(df)} rows, need {window_size})"
            )
            return pd.DataFrame(), []

        features = []
        indices = []

        for start in range(0, len(df) - window_size + 1, step_size):
            window = df.iloc[start:start + window_size]
            feat = self.extract_window_features(window)
            feat["start"] = start
            feat["end"] = start + window_size - 1
            features.append(feat)
            indices.append((start, start + window_size - 1))

        feat_df = pd.DataFrame(features)
        logger.info(f"Extracted {len(feat_df)} windows")

        return feat_df, indices


class IntervalClassifier:
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

    def predict(self, features_df: pd.DataFrame) -> Dict:
        if self.model is None:
            logger.warning("Interval classifier not available")
            return {"available": False, "intervals": []}

        try:
            X = features_df[self.feature_names].fillna(0)
            predictions = self.model.predict(X)

            # Find interval boundaries
            intervals = self._extract_intervals(predictions, features_df)

            logger.info(f"Detected {len(intervals)} intervals")

            return {
                "available": True,
                "intervals": intervals,
                "total_windows": len(predictions),
                "work_windows": int(predictions.sum())
            }

        except Exception as e:
            logger.error(f"Interval prediction failed: {e}")
            return {"available": False, "error": str(e)}

    @staticmethod
    def _extract_intervals(
        predictions: np.ndarray,
        features_df: pd.DataFrame
    ) -> List[Tuple[int, int]]:
        intervals = []
        in_interval = False
        start_idx = 0

        for i, pred in enumerate(predictions):
            if pred == 1 and not in_interval:
                in_interval = True
                start_idx = int(features_df.iloc[i]["start"])
            elif pred == 0 and in_interval:
                in_interval = False
                end_idx = int(features_df.iloc[i - 1]["end"])
                intervals.append((start_idx, end_idx))

        # Handle case where interval extends to end
        if in_interval:
            end_idx = int(features_df.iloc[-1]["end"])
            intervals.append((start_idx, end_idx))

        return intervals


class StrokeQualityPredictor:
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

    def predict(self, features_df: pd.DataFrame) -> Dict:
        if self.model is None:
            logger.warning("Stroke quality predictor not available")
            return {"available": False, "score": None}

        try:
            X = features_df[self.feature_names].fillna(0)
            predictions = self.model.predict(X)

            # Average predictions across windows
            score = float(np.mean(predictions))
            score = max(0.0, min(100.0, score))  # Clip to valid range

            logger.info(f"Predicted stroke quality: {score:.1f}")

            return {
                "available": True,
                "score": round(score, 1),
                "window_scores": {
                    "mean": float(np.mean(predictions)),
                    "std": float(np.std(predictions)),
                    "min": float(np.min(predictions)),
                    "max": float(np.max(predictions))
                }
            }

        except Exception as e:
            logger.error(f"Stroke quality prediction failed: {e}")
            return {"available": False, "error": str(e)}


class AnomalyDetector:
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

    def detect(self, features_df: pd.DataFrame) -> Dict:
        if self.model is None:
            logger.warning("Anomaly detector not available")
            return {"available": False, "anomalies": []}

        try:
            X = features_df[self.feature_names].fillna(0)
            predictions = self.model.predict(X)
            scores = self.model.score_samples(X)

            # Extract anomaly indices (prediction == -1)
            anomaly_indices = []
            for i, pred in enumerate(predictions):
                if pred == -1:
                    anomaly_indices.append(int(features_df.iloc[i]["start"]))

            logger.info(f"Detected {len(anomaly_indices)} anomalies")

            return {
                "available": True,
                "anomalies": anomaly_indices,
                "total_windows": len(predictions),
                "anomaly_count": len(anomaly_indices),
                "anomaly_rate": len(anomaly_indices) / len(predictions),
                "score_statistics": {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores))
                }
            }

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {"available": False, "error": str(e)}


class InferenceEngine:

    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigLoader(config_path)
        self.models_dir = self.config.get('models_dir', 'models')

        # Load models
        loader = ModelLoader(self.models_dir)
        interval_model = loader.load_model('interval_clf')
        sq_model = loader.load_model('stroke_quality_reg')
        anomaly_model = loader.load_model('form_iso')

        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.interval_classifier = IntervalClassifier(
            interval_model,
            self.config.get('interval_features', [])
        )
        self.stroke_quality_predictor = StrokeQualityPredictor(
            sq_model,
            self.config.get('stroke_quality_features', [])
        )
        self.anomaly_detector = AnomalyDetector(
            anomaly_model,
            self.config.get('anomaly_features', [])
        )

        logger.info("Inference engine initialized")

    def analyze(self, df: pd.DataFrame) -> Dict:
        window_size = self.config.get('window_size', 30)
        step_size = self.config.get('step_size', 5)

        # Extract features
        features_df, indices = self.feature_extractor.extract_from_dataframe(
            df, window_size, step_size
        )

        if features_df.empty:
            logger.warning("No features extracted")
            return {"error": "Insufficient data for analysis"}

        # Run all predictions
        results = {
            "intervals": self.interval_classifier.predict(features_df),
            "stroke_quality": self.stroke_quality_predictor.predict(
                features_df
            ),
            "anomalies": self.anomaly_detector.detect(features_df),
            "metadata": {
                "windows_analyzed": len(features_df),
                "window_size": window_size,
                "step_size": step_size,
                "data_length": len(df)
            }
        }

        logger.info("Analysis complete")
        return results


def run_inference(df: pd.DataFrame, config_path: str = "config.yaml") -> Dict:
    engine = InferenceEngine(config_path)
    return engine.analyze(df)