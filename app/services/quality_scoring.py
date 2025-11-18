import pandas as pd
from fuzzywuzzy import fuzz
import re
from app.services.pill_defect_model import detect_pill_defect


import os
os.chdir(r"D:\Projects\PurePill")
os.getcwd()

med_data1 = pd.read_csv("data\A_Z_medicines_dataset_of_India.csv")
med_data2 = pd.read_csv("data\Medicine_Details.csv")

med_data1 = med_data1.rename(columns={
    "name": "medicine_name",
    "manufacturer_name": "manufacturer",
    "short_composition1": "composition_1",
    "short_composition2": "composition_2"
})

med_data2 = med_data2.rename(columns={
    "Medicine Name": "medicine_name",
    "Manufacturer": "manufacturer",
    "Composition": "composition_full"
})

med_data1.drop(axis=1, labels="id", inplace=True)

df = pd.merge(
    med_data1,
    med_data2,
    on=["medicine_name"],
    how="inner",
    suffixes=("_csv1", "_csv2")
)

def clean_medicine_name(name):
    if pd.isna(name):
        return ""

    name = str(name).lower().strip()

    # Remove "tablet", "tab", "capsule", "cap", "syrup", etc.
    name = re.sub(r"\b(tablet|tab|capsule|cap|syrup|injection|mg|ml)\b", "", name)

    # Remove dosage like 500mg, 120 mg, etc.
    name = re.sub(r"\b\d+\s*mg\b", "", name)
    name = re.sub(r"\b\d+\s*ml\b", "", name)

    # Remove bracket dosage like (250 mg)
    name = re.sub(r"\(\s*\d+\s*(mg|ml)\s*\)", "", name)

    # Replace hyphens & underscores with space
    name = name.replace("-", " ").replace("_", " ")

    # Remove duplicate spaces
    name = re.sub(r"\s+", " ", name).strip()

    return name

df["medicine_name"] = df["medicine_name"].apply(clean_medicine_name)

df.drop(labels="manufacturer_csv2", inplace=True, axis=1)

# Calculate review_score vectorized
total = df["Excellent Review %"] + df["Average Review %"] + df["Poor Review %"]

# Avoid division by zero
total = total.replace(0, 1)

df["review_score"] = ((df["Excellent Review %"] * 1) +
                      (df["Average Review %"] * 0.5) +
                      (df["Poor Review %"] * 0)) / total * 100

new_df = df[['medicine_name', 'Is_discontinued', 'manufacturer_csv1', 'type', 'pack_size_label', 'composition_1', 'composition_2', 'composition_full', 'review_score']]

med_info = df[['medicine_name', 'Uses', 'Side_effects']]

canonical_med_data = df[['medicine_name', 'Is_discontinued', 'manufacturer_csv1', 'review_score',
                         'composition_1', 'composition_2', 'composition_full', 'Uses', 'Side_effects']]


# Assuming canonical_med_data is your merged lookup table (loaded once at app startup)

# from pill_defect_model import detect_pill_defect
import pandas as pd
from fuzzywuzzy import fuzz

def calculate_medicine_quality(
    canonical_data: pd.DataFrame,
    ocr_medicine_name: str,
    ocr_manufacturer: str,
    ocr_composition: str,
    pill_image_path: str,       # <-- Pass the pill image path directly
    is_expired: bool,           
    severe_side_effects_check: bool
) -> dict:
    
    # --- Run Pill Defect Detection ---
    pill_result = detect_pill_defect(pill_image_path)
    pill_defect_score = pill_result["defect_score"]  # 0-100
    pill_status = pill_result["similarity_score"] >= 0.50  # Threshold pass/fail
    
    # --- QUALITY WEIGHTS ---
    WEIGHTS = {
        "expiry_status": 0.30,
        "pill_defect_score": 0.20,
        "manufacturer_reliability": 0.20,
        "ingredients_match": 0.20,
        "severe_side_effects": 0.10,
    }

    final_score = 0
    score_details = {}

    # Lookup Drug Information
    canonical_data['match_ratio'] = canonical_data['medicine_name'].apply(
        lambda x: fuzz.ratio(ocr_medicine_name.lower(), x.lower())
    )
    best_match_row = canonical_data[canonical_data['match_ratio'] > 85].sort_values(
        by='match_ratio', ascending=False
    ).head(1)

    if best_match_row.empty:
        return {"Final_Score": 0, "Status": "FAIL", "Details": "Medicine name not found in database."}
    
    med_info = best_match_row.iloc[0]

    # 1. Expiry Status
    expiry_score = 100 if not is_expired else 0
    final_score += expiry_score * WEIGHTS["expiry_status"]
    score_details["Expiry_Status"] = expiry_score

    # 2. Pill Defect Score
    final_score += pill_defect_score * WEIGHTS["pill_defect_score"]
    score_details["Pill_Defect_Score"] = pill_defect_score
    score_details["Pill_Physical_Status"] = "PASS" if pill_status else "FAIL"

    # 3. Manufacturer Reliability
    manufacturer_score = med_info['review_score']
    final_score += manufacturer_score * WEIGHTS["manufacturer_reliability"]
    score_details["Manufacturer_Reliability"] = manufacturer_score

    # 4. Ingredients Match
    known_composition = med_info['composition_full']
    match_ratio = fuzz.partial_ratio(ocr_composition.lower(), known_composition.lower())
    composition_score = match_ratio
    final_score += composition_score * WEIGHTS["ingredients_match"]
    score_details["Ingredients_Match"] = composition_score

    # 5. Severe Side Effects
    side_effects_score = 0 if severe_side_effects_check else 100
    final_score += side_effects_score * WEIGHTS["severe_side_effects"]
    score_details["Severe_Side_Effects"] = side_effects_score

    # Check for discontinued
    if med_info['Is_discontinued']:
        final_score = min(final_score, 10)
        score_details["Warning"] = "Product is officially discontinued."

    final_score = round(final_score / 100, 2) * 100

    # Determine Status
    if not is_expired:
        final_status = "PASS" if final_score >= 60 else "SUSPICIOUS"
    else:
        final_status = "CRITICAL FAIL (EXPIRED)"
        final_score = min(final_score, 5.0)

    return {
        "Final_Score": final_score,
        "Status": final_status,
        "Details": score_details
    }



TEST_DATA = pd.DataFrame({
    'medicine_name': ['Aspirin', 'Amoxicillin', 'Bad Drug X'],
    'Is_discontinued': [False, False, True],
    'manufacturer_csv1': ['Bayer', 'GSK', 'Fake Pharma'],
    'review_score': [95, 80, 20],  # 0-100 score
    'composition_full': ['Acetylsalicylic Acid 500mg', 'Amoxicillin Trihydrate 250mg', 'Banned Compound'],
    'Uses': ['Pain relief', 'Antibiotic', 'Illegal use'],
    'Side_effects': ['Mild nausea', 'Common side effects', 'Severe cardiac arrest risk'],
    'match_ratio': [100, 100, 100] # Adding this column for testing purposes
})

# The test functions you provided:
def test_perfect_pass():
    result = calculate_medicine_quality(
        canonical_data=TEST_DATA, # Pass the data explicitly
        ocr_medicine_name="Aspirin",
        ocr_manufacturer="Bayer",
        ocr_composition="Acetylsalicylic Acid 500mg",
        pill_image_path="app/models/test_pills/pill1.jpg",
        is_expired=False,
        severe_side_effects_check=False
    )
    # The assertions check the output
    assert result['Final_Score'] > 98.0
    assert result['Status'] == 'PASS'
    
def test_expired_fail():
    result = calculate_medicine_quality(
        canonical_data=TEST_DATA,
        ocr_medicine_name="Aspirin",
        ocr_manufacturer="Bayer",
        ocr_composition="Acetylsalicylic Acid 500mg",
        pill_image_path="app/models/test_pills/pill1.jpg",
        is_expired=True, # Critical failure
        severe_side_effects_check=False
    )
    # The assertions check the output
    assert 65.0 < result['Final_Score'] < 70.0
    assert result['Status'] == 'PASS'
    assert result['Details']['Expiry_Status'] == 0


test1 = calculate_medicine_quality(canonical_data=TEST_DATA, # Pass the data explicitly
        ocr_medicine_name="Aspirin",
        ocr_manufacturer="Bayer",
        ocr_composition="Acetylsalicylic Acid 500mg",
        pill_image_path="app/models/test_pills/pill1.jpg",
        is_expired=False,
        severe_side_effects_check=False)

print(test1)