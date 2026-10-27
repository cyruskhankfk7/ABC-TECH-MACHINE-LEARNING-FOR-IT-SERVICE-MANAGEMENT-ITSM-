from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from services.forecast_service import build_monthly_series, forecast_drift, forecast_extra_trees, make_result
from services.model_service import (
    AUTOTAG_FEATURES,
    FORECAST_INFO,
    MODEL_FILES,
    MODEL_METRICS,
    TASK1_FEATURES,
    TASK4_FEATURES,
    ModelStore,
    predict_autotag,
    predict_task1,
    predict_task4,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-this-secret-key-before-production"
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

store = ModelStore(MODEL_DIR)


@app.template_filter("pct")
def pct_filter(value):
    if value is None:
        return "—"
    return f"{float(value) * 100:.2f}%"


@app.template_filter("num")
def num_filter(value):
    if value is None:
        return "—"
    return f"{float(value):,.0f}"


@app.context_processor
def inject_globals():
    return {
        "model_store": store,
        "model_files": MODEL_FILES,
        "task1_features": TASK1_FEATURES,
        "autotag_features": AUTOTAG_FEATURES,
        "task4_features": TASK4_FEATURES,
        "model_metrics": MODEL_METRICS,
        "forecast_info": FORECAST_INFO,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/task1", methods=["GET", "POST"])
def task1():
    result = None
    form_values = request.form.to_dict() if request.method == "POST" else {}
    if request.method == "POST":
        try:
            result = predict_task1(store, form_values)
        except Exception as exc:
            flash(f"Task 1 prediction failed: {exc}", "error")
    return render_template("task1.html", result=result, form_values=form_values)


@app.route("/autotag", methods=["GET", "POST"])
def autotag():
    result = None
    form_values = request.form.to_dict() if request.method == "POST" else {}
    if request.method == "POST":
        try:
            result = predict_autotag(store, form_values)
        except Exception as exc:
            flash(f"Auto-tagging failed: {exc}", "error")
    return render_template("autotag.html", result=result, form_values=form_values)


@app.route("/forecast", methods=["GET", "POST"])
def forecast():
    result = None
    chart_payload = None
    upload_summary = None

    if request.method == "POST":
        uploaded = request.files.get("csv_file")
        model_key = request.form.get("forecast_model", "extra_trees")
        date_format = request.form.get("date_format", "dayfirst")
        horizon = max(1, min(24, int(request.form.get("horizon", 12))))

        if not uploaded or uploaded.filename == "":
            flash("Please choose your incident CSV file.", "error")
            return render_template("forecast.html", result=None, chart_payload=None, upload_summary=None)

        filename = secure_filename(uploaded.filename)
        if not filename.lower().endswith(".csv"):
            flash("Please upload a CSV file.", "error")
            return render_template("forecast.html", result=None, chart_payload=None, upload_summary=None)

        try:
            df = pd.read_csv(uploaded.stream, low_memory=False)
            history = build_monthly_series(df, date_format=date_format)
            if model_key == "drift":
                bundle = store.get("drift")
                forecast_series = forecast_drift(bundle, history, horizon)
                notes = [
                    "Uses the saved Drift slope from the notebook with the latest level observed in the uploaded CSV.",
                    "For an apples-to-apples reproduction of the notebook, upload the same historical period used during training.",
                ]
            elif model_key == "extra_trees":
                bundle = store.get("extra_trees")
                forecast_series = forecast_extra_trees(bundle, history, horizon)
                notes = [
                    "Uses the saved Extra Trees model recursively, starting from the latest monthly history in the uploaded CSV.",
                    "The notebook trained Extra Trees on lagged, rolling and calendar features.",
                ]
            else:
                raise ValueError("Unknown forecasting model selected.")

            result = make_result(history, forecast_series, model_key, notes)
            upload_summary = {
                "filename": filename,
                "rows": len(df),
                "months": len(history),
                "first_month": history.index.min().strftime("%b %Y"),
                "last_month": history.index.max().strftime("%b %Y"),
                "date_format": "DD/MM/YYYY (day-first)" if date_format == "dayfirst" else "MM/DD/YYYY (month-first)",
            }

            chart_payload = {
                "labels": [x.strftime("%b %Y") for x in result.forecast.index],
                "forecast": [round(float(v), 2) for v in result.forecast.values],
            }
        except Exception as exc:
            flash(f"Forecasting failed: {exc}", "error")

    return render_template(
        "forecast.html",
        result=result,
        chart_payload=chart_payload,
        upload_summary=upload_summary,
    )


@app.route("/task4", methods=["GET", "POST"])
def task4():
    result = None
    form_values = request.form.to_dict() if request.method == "POST" else {}
    if request.method == "POST":
        try:
            result = predict_task4(store, form_values)
        except Exception as exc:
            flash(f"Task 4 risk analysis failed: {exc}", "error")
    return render_template("task4.html", result=result, form_values=form_values)


@app.route("/health")
def health():
    return {
        "status": "healthy",
        "application": "ABC Tech ML for ITSM"
    }



@app.errorhandler(413)
def too_large(_error):
    return render_template(
        "error.html",
        title="File too large",
        message="The upload exceeds the 20 MB Flask limit. Your expected ~10 MB CSV is within the configured limit.",
    ), 413

if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 8080))

    app.run(
        debug=False,
        host="0.0.0.0",
        port=port
    )