import numpy as np

def convert_numpy_types(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    return obj


from flask import Flask, request, jsonify, render_template
import os
from app.services.quality_scoring import calculate_medicine_quality
from app.services.ocr_module import run_ocr_pipeline
import pandas as pd

app = Flask(__name__, template_folder="templates")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/check-medicine", methods=["POST"])
def check_medicine():
    medicine_name = request.form.get("medicine_name")
    pill_photo = request.files.get("pill_photo")
    expiry_photo = request.files.get("expiry_photo")

    # Save images temporarily
    pill_path = "temp_pill.jpg"
    expiry_path = "temp_expiry.jpg"
    pill_photo.save(pill_path)
    expiry_photo.save(expiry_path)

    # 1. OCR Check
    ocr_result = run_ocr_pipeline(expiry_path)
    is_expired = ocr_result.get("is_expired", False)

    # Example test data (can replace with your canonical data)
    TEST_DATA = pd.DataFrame({
    'medicine_name': ['Aspirin', 'Amoxicillin', 'Bad Drug X'],
    'Is_discontinued': [False, False, True],
    'manufacturer_csv1': ['Bayer', 'GSK', 'Fake Pharma'],
    'review_score': [95, 80, 20],  # 0-100 score
    'composition_full': ['Acetylsalicylic Acid 500mg', 'Amoxicillin Trihydrate 250mg', 'Banned Compound'],
    'Uses': ['Pain relief', 'Antibiotic', 'Illegal use'],
    'Side_effects': ['Mild nausea', 'Common side effects', 'Severe cardiac arrest risk'],
    'match_ratio': [100, 100, 100]
    })


    # 2. Quality Scoring
    result = calculate_medicine_quality(
        canonical_data=TEST_DATA,   # replace when you use real data
        ocr_medicine_name=medicine_name,
        ocr_manufacturer=None,
        ocr_composition="Acetylsalicylic Acid 500mg",
        pill_image_path=pill_path,
        is_expired=is_expired,
        severe_side_effects_check=False
    )

    clean_result = convert_numpy_types(result)

    return jsonify({
        "success": True,
        "data": clean_result
    })


if __name__ == "__main__":
    app.run(debug=True)
