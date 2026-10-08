from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
import os
import io
import gzip

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "age_clock_model.joblib")
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_names.json")

model = joblib.load(MODEL_PATH)

with open(FEATURE_PATH, "r") as f:
    feature_names = json.load(f)


def read_expression_file(file_storage):

    filename = file_storage.filename.lower()
    raw = file_storage.read()

    if filename.endswith(".gz"):
        raw = gzip.decompress(raw)

    text = raw.decode("utf-8", errors="ignore")

    lines = text.splitlines()

    header_index = None

    for i, line in enumerate(lines):
        if line.startswith("GENEID\t"):
            header_index = i
            break

    if header_index is not None:
        text = "\n".join(lines[header_index:])

    return pd.read_csv(
        io.StringIO(text),
        sep="\t"
    )


@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "model": "AgeClock AI",
        "features": len(feature_names)
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "file" not in request.files:
            return jsonify({
                "error": "Upload an expression file using 'file'."
            }), 400

        uploaded_file = request.files["file"]

        df = read_expression_file(uploaded_file)

        gene_col = None
        expression_col = None

        for col in ["GENEID", "Probe_Name", "ID"]:
            if col in df.columns:
                gene_col = col
                break

        for col in [
            "NORMALIZEDEXPRESSIONVALUE",
            "EXPRESSIONVALUE"
        ]:
            if col in df.columns:
                expression_col = col
                break

        if gene_col is None:
            return jsonify({
                "error": "Probe/GeneID column not found."
            }), 400

        if expression_col is None:
            return jsonify({
                "error": "Expression-value column not found."
            }), 400

        expression_data = df[
            [gene_col, expression_col]
        ].copy()

        expression_data[expression_col] = pd.to_numeric(
            expression_data[expression_col],
            errors="coerce"
        )

        expression_data = expression_data.dropna()

        expression_data = (
            expression_data
            .groupby(gene_col)[expression_col]
            .mean()
        )

        sample = expression_data.reindex(feature_names)

        missing = sample.isna().sum()

        if missing > 0:
            return jsonify({
                "error": "Input is missing required probes.",
                "missing_features": int(missing),
                "required_features": len(feature_names)
            }), 400

        X_input = pd.DataFrame(
            [sample.values],
            columns=feature_names
        )

        predicted_age = float(
            model.predict(X_input)[0]
        )

        result = {
            "predicted_age": round(predicted_age, 2),
            "model": "AgeClock AI",
            "note": "Research/educational prediction, not a medical diagnosis."
        }

        chronological_age = request.form.get(
            "chronological_age"
        )

        if chronological_age:

            chronological_age = float(
                chronological_age
            )

            result["chronological_age"] = chronological_age

            result["age_acceleration"] = round(
                predicted_age - chronological_age,
                2
            )

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )