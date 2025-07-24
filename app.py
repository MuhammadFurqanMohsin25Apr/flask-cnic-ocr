from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR
import re
from datetime import datetime
import logging
import traceback
from difflib import SequenceMatcher

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize RapidOCR
try:
    ocr_engine = RapidOCR()
    logger.info("RapidOCR initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RapidOCR: {str(e)}")
    ocr_engine = None

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_for_cnic(image_path):
    """Optimized preprocessing specifically for CNIC images"""
    try:
        logger.info(f"Preprocessing CNIC image: {image_path}")
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image: {image_path}")
            return [image_path]
        
        # Get image dimensions
        height, width = img.shape[:2]
        logger.info(f"Original dimensions: {width}x{height}")
        
        # Resize for optimal OCR (CNIC cards work best at certain resolutions)
        target_width = 1200
        if width != target_width:
            scale_factor = target_width / width
            new_width = target_width
            new_height = int(height * scale_factor)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            logger.info(f"Resized to: {new_width}x{new_height}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        processed_versions = []
        base_name = image_path.rsplit('.', 1)[0]
        extension = image_path.rsplit('.', 1)[1]
        
        # Version 1: Enhanced contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        enhanced_path = f"{base_name}_enhanced.{extension}"
        cv2.imwrite(enhanced_path, enhanced)
        processed_versions.append(enhanced_path)
        
        # Version 2: Bilateral filter + OTSU threshold
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        _, thresh = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_path = f"{base_name}_thresh.{extension}"
        cv2.imwrite(thresh_path, thresh)
        processed_versions.append(thresh_path)
        
        # Version 3: Gaussian blur + adaptive threshold
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        adaptive_path = f"{base_name}_adaptive.{extension}"
        cv2.imwrite(adaptive_path, adaptive)
        processed_versions.append(adaptive_path)
        
        # Version 4: Sharpened version
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        sharp_path = f"{base_name}_sharp.{extension}"
        cv2.imwrite(sharp_path, sharpened)
        processed_versions.append(sharp_path)
        
        return processed_versions
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return [image_path]

def extract_cnic_number(text):
    """Extract CNIC number in format XXXXX-XXXXXXX-X"""
    if not text:
        return None
    
    # Pattern for CNIC with dashes
    cnic_pattern_with_dashes = r'\b\d{5}-\d{7}-\d{1}\b'
    match = re.search(cnic_pattern_with_dashes, text)
    if match:
        return match.group()
    
    # Pattern for CNIC without dashes (13 consecutive digits)
    cnic_pattern_no_dashes = r'\b\d{13}\b'
    match = re.search(cnic_pattern_no_dashes, text)
    if match:
        cnic = match.group()
        # Format with dashes: XXXXX-XXXXXXX-X
        return f"{cnic[:5]}-{cnic[5:12]}-{cnic[12]}"
    
    # Try to find digits separated by spaces or other characters
    digits_only = re.sub(r'[^\d]', '', text)
    if len(digits_only) == 13:
        return f"{digits_only[:5]}-{digits_only[5:12]}-{digits_only[12]}"
    
    return None

def extract_front_cnic_info(text):
    """Extract information from front side of CNIC"""
    info = {
        'Name': '',
        'Father Name': '',
        'Date of Birth': '',
        'Date of Issue': '',
        'Date of Expiry': '',
        'Gender': '',
        'Country of Stay': '',
        'Identity Number': ''
    }
    
    if not text:
        return info
    
    text_lines = [line.strip() for line in text.split('\n') if line.strip()]
    text_upper = text.upper()
    
    logger.info(f"Processing {len(text_lines)} lines from front image")
    
    # Extract CNIC/Identity Number
    cnic_number = extract_cnic_number(text)
    if cnic_number:
        info['Identity Number'] = cnic_number
    
    # Extract Name
    name_keywords = ['NAME', 'NAM']
    for i, line in enumerate(text_lines):
        line_upper = line.upper()
        for keyword in name_keywords:
            if keyword in line_upper and not info['Name']:
                # Check if name is in the same line
                if ':' in line:
                    name_part = line.split(':', 1)[1].strip()
                    if name_part and len(name_part) > 2 and not any(char.isdigit() for char in name_part):
                        # Clean the name
                        name_clean = re.sub(r'[^\w\s]', '', name_part).strip()
                        if name_clean:
                            info['Name'] = name_clean.title()
                            break
                # Check next line
                elif i + 1 < len(text_lines):
                    next_line = text_lines[i + 1].strip()
                    if next_line and len(next_line) > 2 and not any(char.isdigit() for char in next_line):
                        # Ensure it's not a keyword or system text
                        next_upper = next_line.upper()
                        if not any(kw in next_upper for kw in ['IDENTITY', 'CARD', 'FATHER', 'DATE', 'BIRTH']):
                            name_clean = re.sub(r'[^\w\s]', '', next_line).strip()
                            if name_clean:
                                info['Name'] = name_clean.title()
                                break
        if info['Name']:
            break
    
    # Extract Father's Name
    father_keywords = ['FATHER', 'S/O', 'D/O', 'W/O', 'SON OF', 'DAUGHTER OF']
    for i, line in enumerate(text_lines):
        line_upper = line.upper()
        for keyword in father_keywords:
            if keyword in line_upper and not info['Father Name']:
                # Check if father name is in the same line
                if ':' in line:
                    father_part = line.split(':', 1)[1].strip()
                    if father_part and len(father_part) > 2 and not any(char.isdigit() for char in father_part):
                        father_clean = re.sub(r'[^\w\s]', '', father_part).strip()
                        if father_clean:
                            info['Father Name'] = father_clean.title()
                            break
                # Check next line
                elif i + 1 < len(text_lines):
                    next_line = text_lines[i + 1].strip()
                    if next_line and len(next_line) > 2 and not any(char.isdigit() for char in next_line):
                        next_upper = next_line.upper()
                        if not any(kw in next_upper for kw in ['IDENTITY', 'CARD', 'DATE', 'BIRTH']):
                            father_clean = re.sub(r'[^\w\s]', '', next_line).strip()
                            if father_clean:
                                info['Father Name'] = father_clean.title()
                                break
        if info['Father Name']:
            break
    
    # Extract Gender
    gender_patterns = [
        r'\b(MALE|M)\b',
        r'\b(FEMALE|F)\b',
        r'\b(X)\b'
    ]
    for pattern in gender_patterns:
        match = re.search(pattern, text_upper)
        if match:
            gender = match.group(1)
            if gender in ['MALE']:
                info['Gender'] = 'M'
            elif gender in ['FEMALE']:
                info['Gender'] = 'F'
            else:
                info['Gender'] = gender
            break
    
    # Extract dates in format XX.XX.XXXX
    date_pattern = r'\b(\d{1,2}[./-]\d{1,2}[./-]\d{4})\b'
    dates_found = re.findall(date_pattern, text)
    
    # Convert dates to XX.XX.XXXX format
    formatted_dates = []
    for date in dates_found:
        # Replace any separator with dot
        formatted_date = re.sub(r'[/-]', '.', date)
        # Ensure two digits for day and month
        parts = formatted_date.split('.')
        if len(parts) == 3:
            day, month, year = parts
            formatted_date = f"{day.zfill(2)}.{month.zfill(2)}.{year}"
            formatted_dates.append(formatted_date)
    
    # Assign dates based on context and position
    if formatted_dates:
        # Look for specific date contexts
        birth_found = False
        issue_found = False
        expiry_found = False
        
        for i, line in enumerate(text_lines):
            line_upper = line.upper()
            if 'BIRTH' in line_upper or 'DOB' in line_upper:
                for date in formatted_dates:
                    if date in line or any(part in line for part in date.split('.')):
                        info['Date of Birth'] = date
                        birth_found = True
                        break
            elif 'ISSUE' in line_upper in line_upper:
                for date in formatted_dates:
                    if date in line or any(part in line for part in date.split('.')):
                        info['Date of Issue'] = date
                        issue_found = True
                        break
            elif 'EXPIRY' in line_upper or 'EXPIRE' in line_upper or 'VALID' in line_upper:
                for date in formatted_dates:
                    if date in line or any(part in line for part in date.split('.')):
                        info['Date of Expiry'] = date
                        expiry_found = True
                        break
        
        # If context-based assignment didn't work, assign by typical order
        remaining_dates = [d for d in formatted_dates]
        if not birth_found and remaining_dates:
            # Birth date is usually the earliest year
            birth_date = min(remaining_dates, key=lambda x: int(x.split('.')[2]))
            info['Date of Birth'] = birth_date
            remaining_dates.remove(birth_date)
        
        if not issue_found and remaining_dates:
            info['Date of Issue'] = remaining_dates[0]
            remaining_dates.remove(remaining_dates[0])
        
        if not expiry_found and remaining_dates:
            info['Date of Expiry'] = remaining_dates[0]
    
    # Extract Country of Stay
    country_keywords = ['PAKISTAN', 'PAK']
    for keyword in country_keywords:
        if keyword in text_upper:
            info['Country of Stay'] = 'Pakistan'
            break
    
    # If no country found, look for any country-like text
    if not info['Country of Stay']:
        country_pattern = r'\b[A-Z]{2,}[A-Z\s]{2,}\b'
        countries = re.findall(country_pattern, text_upper)
        for country in countries:
            if len(country.strip()) > 3 and country.strip() not in ['IDENTITY', 'CARD', 'NUMBER']:
                info['Country of Stay'] = country.strip().title()
                break
    
    return info

def extract_back_cnic_info(text):
    """Extract information from back side of CNIC"""
    info = {
        'Identity Number': '',
    }
    
    if not text:
        return info
    
    # Extract CNIC number
    cnic_number = extract_cnic_number(text)
    if cnic_number:
        info['Identity Number'] = cnic_number
    
    
    return info

def match_cnic_numbers(front_info, back_info):
    """Compare CNIC numbers from front and back images"""
    front_cnic = front_info.get('Identity Number', '')
    back_cnic = back_info.get('Identity Number', '')
    
    if not front_cnic or not back_cnic:
        return {
            'match': False,
            'confidence': 0.0,
            'front_cnic': front_cnic,
            'back_cnic': back_cnic,
            'message': 'CNIC number not found in one or both images'
        }
    
    # Direct match
    if front_cnic == back_cnic:
        return {
            'match': True,
            'confidence': 1.0,
            'front_cnic': front_cnic,
            'back_cnic': back_cnic,
            'message': 'CNIC numbers match perfectly'
        }
    
    # Calculate similarity
    similarity = SequenceMatcher(None, front_cnic, back_cnic).ratio()
    
    return {
        'match': similarity > 0.9,  # 90% similarity threshold
        'confidence': similarity,
        'front_cnic': front_cnic,
        'back_cnic': back_cnic,
        'message': f'CNIC numbers similarity: {similarity:.2%}'
    }

def perform_ocr(image_path):
    """Perform OCR on image and return best result"""
    if ocr_engine is None:
        return ""
    
    try:
        # Try original image first
        original_text = ""
        result = ocr_engine(image_path)
        if result and result[0]:
            original_text = "\n".join([item[1] for item in result[0]])
        
        # Try preprocessed versions
        processed_paths = preprocess_image_for_cnic(image_path)
        processed_texts = []
        
        for processed_path in processed_paths:
            try:
                result = ocr_engine(processed_path)
                if result and result[0]:
                    text = "\n".join([item[1] for item in result[0]])
                    if text.strip():
                        processed_texts.append(text)
            except Exception as e:
                logger.warning(f"OCR failed for {processed_path}: {str(e)}")
        
        # Clean up processed files
        for processed_path in processed_paths:
            try:
                if os.path.exists(processed_path):
                    os.remove(processed_path)
            except:
                pass
        
        # Return the longest text (usually most complete)
        all_texts = [original_text] + processed_texts
        all_texts = [t for t in all_texts if t.strip()]
        
        if all_texts:
            return max(all_texts, key=len)
        return ""
        
    except Exception as e:
        logger.error(f"OCR failed: {str(e)}")
        return ""

def format_cnic_output(front_info, back_info, cnic_match):
    """Format the final output"""
    output_lines = []
    
    # Front information
    if front_info.get('Name'):
        output_lines.append(f"Name: {front_info['Name']}")
    if front_info.get('Father Name'):
        output_lines.append(f"Father Name: {front_info['Father Name']}")
    if front_info.get('Date of Birth'):
        output_lines.append(f"Date of Birth: {front_info['Date of Birth']}")
    if front_info.get('Date of Issue'):
        output_lines.append(f"Date of Issue: {front_info['Date of Issue']}")
    if front_info.get('Date of Expiry'):
        output_lines.append(f"Date of Expiry: {front_info['Date of Expiry']}")
    if front_info.get('Gender'):
        output_lines.append(f"Gender: {front_info['Gender']}")
    if front_info.get('Country of Stay'):
        output_lines.append(f"Country of Stay: {front_info['Country of Stay']}")
    if front_info.get('Identity Number'):
        output_lines.append(f"Identity Number: {front_info['Identity Number']}")
    
    
    # CNIC matching information
    output_lines.append(f"\nCNIC Validation: {cnic_match['message']}")
    
    return '\n'.join(output_lines)

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'message': 'Enhanced CNIC OCR Backend is running',
        'status': 'active',
        'ocr_engine_status': 'initialized' if ocr_engine else 'failed',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/extract-cnic', methods=['POST'])
def extract_cnic():
    """Extract and validate CNIC information from front and back images"""
    try:
        logger.info("CNIC extraction endpoint called")
        
        if ocr_engine is None:
            return jsonify({'error': 'OCR engine not initialized'}), 500
        
        # Check for required files
        if 'front_image' not in request.files or 'back_image' not in request.files:
            return jsonify({
                'error': 'Both front_image and back_image are required'
            }), 400
        
        front_file = request.files['front_image']
        back_file = request.files['back_image']
        
        if front_file.filename == '' or back_file.filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        if not (allowed_file(front_file.filename) and allowed_file(back_file.filename)):
            return jsonify({
                'error': 'Invalid file format. Allowed formats: png, jpg, jpeg, gif, bmp'
            }), 400
        
        # Save files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        front_filename = secure_filename(f"front_{timestamp}_{front_file.filename}")
        back_filename = secure_filename(f"back_{timestamp}_{back_file.filename}")
        
        front_path = os.path.join(app.config['UPLOAD_FOLDER'], front_filename)
        back_path = os.path.join(app.config['UPLOAD_FOLDER'], back_filename)
        
        front_file.save(front_path)
        back_file.save(back_path)
        
        logger.info(f"Files saved: {front_path}, {back_path}")
        
        # Perform OCR
        front_text = perform_ocr(front_path)
        back_text = perform_ocr(back_path)
        
        logger.info(f"Front OCR length: {len(front_text)}, Back OCR length: {len(back_text)}")
        
        # Extract structured information
        front_info = extract_front_cnic_info(front_text)
        back_info = extract_back_cnic_info(back_text)
        
        # Match CNIC numbers
        cnic_match = match_cnic_numbers(front_info, back_info)
        
        # Use front CNIC if back CNIC is empty but front has it
        if not back_info['Identity Number'] and front_info.get('Identity Number'):
            back_info['Identity Number'] = front_info['Identity Number']
        
        final_info = {
            **front_info,
        }
        
        # Clean up uploaded files
        try:
            os.remove(front_path)
            os.remove(back_path)
        except:
            pass
        
        return jsonify({
            'success': True,
            'extracted_data': final_info,
            'cnic_validation': cnic_match,
            'formatted_output': format_cnic_output(front_info, back_info, cnic_match),
            'raw_text': {
                'front': front_text,
                'back': back_text
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing CNIC extraction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/extract-single', methods=['POST'])
def extract_single():
    """Extract information from a single CNIC image"""
    try:
        if 'image' not in request.files or 'side' not in request.form:
            return jsonify({
                'error': 'Both image file and side parameter (front/back) are required'
            }), 400
        
        image_file = request.files['image']
        side = request.form['side'].lower()
        
        if side not in ['front', 'back']:
            return jsonify({'error': 'Side must be either "front" or "back"'}), 400
        
        if image_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(image_file.filename):
            return jsonify({'error': 'Invalid file format'}), 400
        
        # Save file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(f"{side}_{timestamp}_{image_file.filename}")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(file_path)
        
        # Perform OCR
        extracted_text = perform_ocr(file_path)
        
        # Extract information based on side
        if side == 'front':
            extracted_info = extract_front_cnic_info(extracted_text)
        else:
            extracted_info = extract_back_cnic_info(extracted_text)
        
        # Clean up
        try:
            os.remove(file_path)
        except:
            pass
        
        return jsonify({
            'success': True,
            'side': side,
            'extracted_data': extracted_info,
            'raw_text': extracted_text,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing single image: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=5000)