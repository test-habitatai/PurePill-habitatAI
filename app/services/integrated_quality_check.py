# app/services/integrated_quality_check.py

from ocr_module import run_ocr_pipeline
from quality_scoring import calculate_medicine_quality
import pandas as pd

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

# Paths to images
pill_image_path = "app/models/test_pills/pill1.jpg"     # For defect detection
ocr_image_path  = "app/services/med_check3.jpg" # For OCR expiry check

# Step 1: Run OCR pipeline to determine expiry
ocr_result = run_ocr_pipeline(ocr_image_path)

if not ocr_result['success']:
    print(f"OCR failed: {ocr_result.get('error')}")
    is_expired = False  # Fallback
else:
    is_expired = ocr_result['is_expired']
    print(f"Medicine Expired? {is_expired}")
    print(f"Raw OCR Text: {ocr_result.get('raw_ocr_text')}\n")

# Step 2: Run quality scoring using OCR result and pill image
quality_result = calculate_medicine_quality(
    canonical_data=TEST_DATA,
    ocr_medicine_name="Aspirin",      # Replace with your OCR-detected medicine name if you have an extraction step
    ocr_manufacturer="Bayer",          # Replace similarly if extracted from OCR
    ocr_composition="Acetylsalicylic Acid 500mg", # Replace with extracted composition
    pill_image_path=pill_image_path,
    is_expired=is_expired,
    severe_side_effects_check=False
)

print("\n--- Final Quality Result ---")
print(quality_result)
