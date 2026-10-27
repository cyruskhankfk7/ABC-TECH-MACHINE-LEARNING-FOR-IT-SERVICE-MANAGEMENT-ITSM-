from __future__ import annotations
import xgboost as xgb
from sklearn.pipeline import Pipeline
import __main__
from services.custom_transformers import FrequencyEncoder

# The saved Task 4 models were pickled with FrequencyEncoder
# defined in the notebook's __main__ namespace.
setattr(__main__, "FrequencyEncoder", FrequencyEncoder)

import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


TASK1_FEATURES = [
    "CI_Cat",
    "CI_Subcat",
    "WBS",
    "Impact",
    "Urgency",
    "number_cnt",
    "Category",
    "KB_number",
    "Alert_Status",
    "No_of_Related_Interactions",
    "Open_Year",
    "Open_Month",
    "Open_DayOfWeek",
    "Open_Hour",
    "Is_Weekend",
]

TASK1_NUMERIC = {
    "Impact",
    "Urgency",
    "number_cnt",
    "No_of_Related_Interactions",
    "Open_Year",
    "Open_Month",
    "Open_DayOfWeek",
    "Open_Hour",
    "Is_Weekend",
}

AUTOTAG_FEATURES = [
    "CI_Subcat",
    "WBS",
    "Category",
    "Impact",
    "Urgency",
    "Open_Hour",
    "Open_DayOfWeek",
    "Open_Month",
    "Open_Quarter",
    "Open_IsWeekend",
]

AUTOTAG_NUMERIC = {
    "Impact",
    "Urgency",
    "Open_Hour",
    "Open_DayOfWeek",
    "Open_Month",
    "Open_Quarter",
    "Open_IsWeekend",
}

TASK4_FEATURES = [
    "CI_Name",
    "CI_Cat",
    "CI_Subcat",
    "CI_Prefix",
    "WBS",
    "WBS_Prefix",
    "Impact",
    "Urgency",
    "Priority",
    "number_cnt",
    "Category",
    "Alert_Status",
    "Open_Year",
    "Open_Month",
    "Open_Day",
    "Open_DayOfWeek",
    "Open_Hour",
    "Open_Quarter",
    "Open_WeekOfYear",
    "Is_Weekend",
    "Hour_Sin",
    "Hour_Cos",
    "Month_Sin",
    "Month_Cos",
    "DayOfWeek_Sin",
    "DayOfWeek_Cos",
    "CI_Prior_Incident_Count",
    "CI_Prior_RFC_Count",
    "CI_Prior_RFC_Rate",
    "CI_Prior_Change_Count",
    "CI_Prior_Failure_Count",
    "CI_Prior_Failure_Rate",
    "CI_Days_Since_Last_Incident",
    "CI_Incidents_30D",
    "CI_Incidents_90D",
    "CI_RFC_90D",
    "CI_Failure_90D",
    "CI_Prior_Avg_Impact",
    "CI_Prior_Avg_Urgency",
    "Category_Prior_Incident_Count",
    "Category_Prior_RFC_Count",
    "Category_Prior_RFC_Rate",
    "WBS_Prior_Incident_Count",
    "WBS_Prior_RFC_Count",
    "WBS_Prior_RFC_Rate",
]


MODEL_FILES = {
    "task1": "task1_xgb_classifier.json",
    "priority_autotag": "ABC_Tech_Priority_AutoTag_Model.pkl",
    "department_autotag": "ABC_Tech_Department_AutoTag_Model.pkl",
    "rfc": "rfc_prediction_model.pkl",
    "asset": "asset_failure_risk_model.pkl",
    "drift": "abc_tech_drift_model.joblib",
    "extra_trees": "abc_tech_extra_trees_model.joblib",
}

TASK1_PREPROCESSOR_FILE = "task1_preprocessor.joblib"

MODEL_METRICS = {
    "task1": {
        "model": "XGBoost",
        "metrics": {
            "Accuracy": 0.9999,
            "Precision": 0.9929,
            "Recall": 1.0000,
            "F1": 0.9964,
            "ROC-AUC": 1.0000,
        },
        "note": "Holdout metrics reported in the notebook.",
    },
    "priority_autotag": {
        "model": "Random Forest",
        "metrics": {
            "Accuracy": 0.9991,
            "Macro Precision": 0.9988,
            "Macro Recall": 0.9927,
            "Macro F1": 0.9957,
            "Weighted F1": 0.9991,
        },
        "note": "Selected priority auto-tagging model from the notebook.",
    },
    "department_autotag": {
        "model": "Logistic Regression",
        "metrics": {
            "Accuracy": 0.8898,
            "Macro Precision": 0.9624,
            "Macro Recall": 0.9825,
            "Macro F1": 0.9681,
            "Weighted F1": 0.8973,
        },
        "note": "CI_Cat is used as the department/configuration-category proxy.",
    },
    "rfc": {
        "model": "Extra Trees",
        "metrics": {
            "Accuracy": 0.9886,
            "Precision": 0.5078,
            "Recall": 0.6019,
            "F1": 0.5508,
            "ROC-AUC": 0.8979,
            "PR-AUC": 0.4957,
            "Threshold": 0.31,
        },
        "note": "Threshold was selected from validation and retained for test/production inference.",
    },
    "asset": {
        "model": "Logistic Regression",
        "metrics": {
            "Accuracy": 0.9669,
            "Precision": 0.9796,
            "Recall": 0.9842,
            "F1": 0.9819,
            "ROC-AUC": 0.9872,
            "PR-AUC": 0.9988,
            "Threshold": 0.06,
        },
        "note": "Threshold was selected from validation and retained for test/production inference.",
    },
}


FORECAST_INFO = {
    "drift": {
        "name": "Drift",
        "rolling_cv": {
            "WAPE": 68.21,
            "MAE": 1044.9,
            "RMSE": 1659.4,
        },
        "holdout": {
            "MAE": 857.333,
            "RMSE": 1042.440,
            "R2": -0.732,
            "WAPE": 11.702,
            "sMAPE": 11.745,
            "MASE": 1.798,
        },
        "strengths": [
            "Very simple and highly interpretable.",
            "Extremely fast to train and forecast.",
            "Requires little computational infrastructure.",
            "Captures overall long-term trend.",
            "Useful as a baseline for comparing more complex models.",
        ],
        "weaknesses": [
            "Does not explicitly model seasonality.",
            "Does not capture short-term nonlinear changes.",
            "Can perform poorly when volumes fluctuate irregularly.",
            "Assumes a historical trend continues into the future.",
            "Rolling-CV WAPE of 68.21% indicates unstable historical performance.",
        ],
    },
    "extra_trees": {
        "name": "Extra Trees",
        "holdout": {
            "MAE": 574.339,
            "RMSE": 715.258,
            "R2": 0.184,
            "WAPE": 7.839,
            "sMAPE": 7.766,
            "MASE": 1.205,
        },
        "strengths": [
            "Captures nonlinear relationships between lagged volumes.",
            "Learns interactions among lagged, rolling and calendar features.",
            "More flexible than a trend-only model.",
            "Performed strongly on the current unseen holdout period.",
            "Can improve short-term prediction when the recent history is informative.",
        ],
        "weaknesses": [
            "Less interpretable than Drift.",
            "Depends heavily on lag and rolling features.",
            "Recursive forecasting can accumulate prediction errors.",
            "Performance may decline when future patterns differ materially from history.",
            "Requires periodic retraining as new history arrives.",
        ],
    },
}


class ModelStore:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.models: dict[str, Any] = {}
        self.load_errors: dict[str, str] = {}
        self._load_all()

    def _load_all(self) -> None:
        for key, filename in MODEL_FILES.items():
            path = self.model_dir / filename

            if not path.exists():
                self.load_errors[key] = f"Missing file: {filename}"
                continue

            try:
                if key == "task1":
                    # Load the saved preprocessing pipeline
                    preprocessor_path = (
                            self.model_dir / TASK1_PREPROCESSOR_FILE
                    )

                    if not preprocessor_path.exists():
                        raise FileNotFoundError(
                            f"Missing Task 1 preprocessor: "
                            f"{TASK1_PREPROCESSOR_FILE}"
                        )

                    preprocessor = joblib.load(
                        preprocessor_path
                    )

                    # Load XGBoost using the stable JSON format
                    classifier = xgb.XGBClassifier()
                    classifier.load_model(path)

                    # Rebuild the original two-step pipeline
                    obj = Pipeline(
                        steps=[
                            ("preprocessor", preprocessor),
                            ("classifier", classifier),
                        ]
                    )

                elif key in {"rfc", "asset"}:
                    with path.open("rb") as fh:
                        obj = pickle.load(fh)

                else:
                    obj = joblib.load(path)

                self.models[key] = obj

            except Exception as exc:
                self.load_errors[key] = (
                    f"Could not load {filename}: {exc}"
                )
    def get(self, key: str) -> Any:
        if key not in self.models:
            raise RuntimeError(self.load_errors.get(key, f"Model '{key}' is not loaded."))
        return self.models[key]

    def has(self, key: str) -> bool:
        return key in self.models


def _frame_from_mapping(mapping: dict[str, Any], features: list[str], numeric: set[str]) -> pd.DataFrame:
    row = {}
    for feature in features:
        value = mapping.get(feature, "")
        if value == "":
            value = np.nan
        if feature in numeric:
            row[feature] = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        else:
            row[feature] = value
    return pd.DataFrame([row], columns=features)


def prepare_task1_input(form: dict[str, Any]) -> pd.DataFrame:
    return _frame_from_mapping(form, TASK1_FEATURES, TASK1_NUMERIC)


def prepare_autotag_input(form: dict[str, Any]) -> pd.DataFrame:
    return _frame_from_mapping(form, AUTOTAG_FEATURES, AUTOTAG_NUMERIC)


def prepare_task4_input(form: dict[str, Any], feature_types: dict[str, Any] | None = None) -> pd.DataFrame:
    numeric = set(feature_types.get("numeric", [])) if feature_types else set()
    if not numeric:
        numeric = {
            f for f in TASK4_FEATURES
            if any(token in f.lower() for token in [
                "count", "rate", "days", "incidents", "rfc", "failure", "impact",
                "urgency", "number_cnt", "priority", "year", "month", "day", "hour",
                "quarter", "weekofyear", "week", "sin", "cos", "is_weekend"
            ])
        }
    return _frame_from_mapping(form, TASK4_FEATURES, numeric)


def probability_for_positive(model: Any, X: pd.DataFrame) -> float | None:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if np.ndim(proba) == 2 and proba.shape[1] >= 2:
            classes = getattr(model, "classes_", [0, 1])
            try:
                positive_idx = list(classes).index(1)
            except ValueError:
                positive_idx = proba.shape[1] - 1
            return float(proba[0, positive_idx])
    return None


def predict_task1(store: ModelStore, form: dict[str, Any]) -> dict[str, Any]:
    model = store.get("task1")
    X = prepare_task1_input(form)
    prediction = int(model.predict(X)[0])
    probability = probability_for_positive(model, X)
    return {
        "prediction": prediction,
        "label": "High Priority (P1/P2)" if prediction == 1 else "Normal Priority (P3/P4/P5)",
        "probability": probability,
        "model": "XGBoost",
        "metrics": MODEL_METRICS["task1"],
    }


def _prediction_confidence(model: Any, X: pd.DataFrame) -> float | None:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        return float(np.max(probs))
    return None


def predict_autotag(store: ModelStore, form: dict[str, Any]) -> dict[str, Any]:
    X = prepare_autotag_input(form)
    p_model = store.get("priority_autotag")
    d_model = store.get("department_autotag")

    priority = p_model.predict(X)[0]
    department = d_model.predict(X)[0]

    return {
        "priority": int(priority) if isinstance(priority, (int, np.integer, float, np.floating)) else priority,
        "priority_confidence": _prediction_confidence(p_model, X),
        "department": department,
        "department_confidence": _prediction_confidence(d_model, X),
        "priority_model": "Random Forest",
        "department_model": "Logistic Regression",
        "priority_metrics": MODEL_METRICS["priority_autotag"],
        "department_metrics": MODEL_METRICS["department_autotag"],
    }


def predict_task4(store: ModelStore, form: dict[str, Any]) -> dict[str, Any]:
    rfc_package = store.get("rfc")
    asset_package = store.get("asset")

    rfc_model = rfc_package["model"]
    asset_model = asset_package["model"]

    feature_types = rfc_package.get("feature_types") or asset_package.get("feature_types")
    X = prepare_task4_input(form, feature_types=feature_types)

    rfc_probability = probability_for_positive(rfc_model, X)
    asset_probability = probability_for_positive(asset_model, X)

    rfc_threshold = float(rfc_package.get("threshold", MODEL_METRICS["rfc"]["metrics"]["Threshold"]))
    asset_threshold = float(asset_package.get("threshold", MODEL_METRICS["asset"]["metrics"]["Threshold"]))

    rfc_pred = int(rfc_probability >= rfc_threshold) if rfc_probability is not None else int(rfc_model.predict(X)[0])
    asset_pred = int(asset_probability >= asset_threshold) if asset_probability is not None else int(asset_model.predict(X)[0])

    return {
        "rfc_prediction": rfc_pred,
        "rfc_label": "RFC Required" if rfc_pred == 1 else "No RFC Signal",
        "rfc_probability": rfc_probability,
        "rfc_threshold": rfc_threshold,
        "rfc_model": rfc_package.get("model_name", "Extra Trees"),
        "rfc_metrics": MODEL_METRICS["rfc"],
        "asset_prediction": asset_pred,
        "asset_label": "Future Failure / Misconfiguration Risk Signal" if asset_pred == 1 else "No Future Risk Signal",
        "asset_probability": asset_probability,
        "asset_threshold": asset_threshold,
        "asset_model": asset_package.get("model_name", "Logistic Regression"),
        "asset_metrics": MODEL_METRICS["asset"],
        "horizon_days": asset_package.get("horizon_days", 90),
    }
