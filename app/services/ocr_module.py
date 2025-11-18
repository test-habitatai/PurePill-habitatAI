# app/services/ocr_module.py

import easyocr
import re
from typing import Dict, Any
from datetime import datetime
import cv2 # For image loading (Ensure 'opencv-python' is installed)
import numpy as np
import os # Added for path handling

# Initialize EasyOCR Reader once
try:
    # Use English ('en') and disable GPU if memory is an issue
    READER = easyocr.Reader(['en'], gpu=False)
except Exception as e:
    # This print will appear on startup if initialization fails
    print(f"Error initializing EasyOCR: {e}. Ensure torch and dependencies are installed.")
    READER = None

def extract_expiry_status(ocr_text: str) -> bool:
    """
    Scans the OCR text for common date patterns (MM/YY or MM/YYYY)
    and determines if the medicine is expired relative to the current date (Nov 2025).
    Returns True if expired, False otherwise.
    """
    # 🚨 NEW: Clean the text before running regex
    # Replace common OCR noise that separates date labels (like underscore or extra space)
    cleaned_text = ocr_text.upper().replace('_', ' ').replace('.', '/')
    # NOTE: datetime.now().date() is the current time (Nov 18, 2025)
    today = datetime.now().date()
    
    # Simple regex to capture date patterns and keywords (EXPIRY DATE, EXP, etc.)
    date_patterns = [
    # 1. NEW & IMPROVED: Highly permissive pattern for labeled textual dates.
    # It now allows 3-4 alphabetical characters [A-Z]{3,4} for the month
    # and handles zero or more separators between EXP and the date.
    r'(?:EXPIRY|EXPIRES|EXP)\s*[:\s/\-]*([A-Z]{3,4}[/\-]\d{2,4})',
    
    # 2. Numeric Dates (MM/YY, MM/YYYY) - Unchanged
    r'(\d{1,2})[/\-](\d{2,4})', 
    
    # 3. Labeled Numeric Dates - Unchanged
    r'(?:EXPIRY\s*DATE|EXPIRES|EXP)\s*[:\s/\-]*(\d{1,2}[/\-]\d{2,4})'
]
    
    potential_dates = []
    
    for pattern in date_patterns:
        # Use the cleaned text for matching
        matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
        if matches:
            print(f"DEBUG: Pattern '{pattern[:25]}...' matched: {matches}") # <--- ADD THIS LINE
        for match in matches:
            # Match is typically (month, year) tuple or a full date string if using the label group
            if isinstance(match, tuple):
                m, y = match
                potential_dates.append(f"{m}/{y}")
            elif isinstance(match, str) and match:
                potential_dates.append(match)


    for date_str in potential_dates:
        try:
            date_str = date_str.replace(' ', '').upper().replace('.', '/') # Clean/Normalize
            
            year_part = date_str.split('/')[-1].split('-')[-1]
            
            # 1. Check if the date starts with an alphabetical month (e.g., JUN/25)
            if year_part.isdigit() and not date_str.split('/')[0].isdigit():
                # Format: MMM/YY or MMM/YYYY (e.g., JUN/25 or JUN/2025)
                # Use %b for abbreviated month name and %y or %Y for year
                date_format = '%b/%y' if len(year_part) == 2 else '%b/%Y'
                exp_date = datetime.strptime(date_str, date_format).date().replace(day=1) 

            # 2. Check for numeric month/year (your existing logic)
            elif year_part.isdigit():
                # Format: MM/YY or MM/YYYY (e.g., 06/25 or 06/2025)
                date_format = '%m/%y' if len(year_part) == 2 else '%m/%Y'
                exp_date = datetime.strptime(date_str, date_format).date().replace(day=1)
            else:
                continue # Not a recognized format
                
            # Check for expiry against today's date
            if exp_date < today:
                return True # CRITICAL: EXPIRED
        
        except ValueError:
            continue
            
    # ...
            
    return False # Not expired (or no date found, which is treated as not expired for MVP)

def run_ocr_pipeline(image_path: str) -> Dict[str, Any]:
    """
    Main function to run OCR and extract ONLY the expiry status.
    """
    if not READER:
        return {"success": False, "error": "OCR Reader not initialized."}
        
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")
            
        # Run OCR to get all text
        results = READER.readtext(img, detail=0, paragraph=True)
        full_text = " ".join(results)
        print(f"\n--- DEBUG: Full OCR Text ---")
        print(full_text) # <--- ADD THIS LINE
        print("---------------------------")
        
        # 1. Expiry Status (The only required output)
        is_expired = extract_expiry_status(full_text)
        
        return {
            "is_expired": is_expired,
            "success": True,
            # TEMPORARILY return the raw text to confirm what was passed to the function
            "raw_ocr_text": full_text
        }
        
    except FileNotFoundError as f:
        return {"success": False, "error": str(f)}
    except Exception as e:
        return {"success": False, "error": f"OCR processing failed: {e}"}

# --- Example of how to call this function for testing ---
if __name__ == '__main__':
    # NOTE: Replace 'medicine_test_2.png' with a real path for testing.
    # Ensure you are running this from the correct directory or use an absolute path.
    test_image_path = os.path.join(os.getcwd(), 'med_check3.jpg') 
    
    print(f"Testing OCR on: {test_image_path}")
    
    if not os.path.exists(test_image_path):
        print("\n*** WARNING ***: Test image not found. Create a dummy image or change the path to test.")
    else:
        result = run_ocr_pipeline(test_image_path)
        print("\n--- OCR Result ---")
        print(result)