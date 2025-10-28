# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.responses import JSONResponse, HTMLResponse
# from fastapi.middleware.cors import CORSMiddleware
# from io import BytesIO
# import base64
# import json
# import os
# from dotenv import load_dotenv
# import logging
# import re
# import pandas as pd
# from openai import OpenAI
# from pathlib import Path

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()

# # Load OpenAI API key
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     logger.error("⚠️ OPENAI_API_KEY environment variable not set")
#     raise RuntimeError("OPENAI_API_KEY environment variable not set")

# # Initialize OpenAI client
# try:
#     client = OpenAI(api_key=OPENAI_API_KEY)
#     logger.info("✅ OpenAI client initialized successfully")
# except Exception as e:
#     logger.error(f"❌ Failed to initialize OpenAI client: {str(e)}")
#     raise RuntimeError(f"Failed to initialize OpenAI client: {str(e)}")

# app = FastAPI(title="Insurance Policy Processing System")

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Simplified Formula Data - Only for ICICI

# FORMULA_DATA = [
#     {"LOB": "TW", "SEGMENT": "1+5", "INSURER": "TATA", "PO": "90% of Payin", "REMARKS": "NIL"},

#     {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "TATA", "PO": "90% of Payin", "REMARKS": "NIL"},



    
#     {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "TATA", "PO": "-2%", "REMARKS": "Payin Below 20%"},
#     {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "TATA", "PO": "-3%", "REMARKS": "Payin 21% to 30%"},
#     {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "TATA", "PO": "-4%", "REMARKS": "Payin 31% to 50%"},
#     {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "TATA", "PO": "-5%", "REMARKS": "Payin Above 50%"},


#     {"LOB": "PVT CAR", "SEGMENT": "PVT CAR COMP + SAOD", "INSURER": "TATA", "PO": "90% of Payin", "REMARKS": "Payin Below 20"},
#     {"LOB": "PVT CAR", "SEGMENT": "PVT CAR COMP + SAOD", "INSURER": "TATA", "PO": "90% of Payin", "REMARKS": "Payin Above 20%"},
#     {"LOB": "PVT CAR", "SEGMENT": "PVT CAR COMP + SAOD", "INSURER": "TATA", "PO": "90% of Payin", "REMARKS": "Payin Above 30%"},
#     {"LOB": "PVT CAR", "SEGMENT": "PVT CAR COMP + SAOD", "INSURER": "TATA", "PO": "90% of Payin", "REMARKS": "Payin Above 40%"},
#     {"LOB": "PVT CAR", "SEGMENT": "PVT CAR COMP + SAOD", "INSURER": "TATA", "PO": "90% of Payin", "REMARKS": "Payin Above 50%"},

#     {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "TATA", "PO": "90% of Payin", "REMARKS": "Payin Below 20%"},
#       {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "TATA", "PO": "90% of Payin", "REMARKS": "Payin Above 20%"},
#         {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "TATA", "PO": "90% of Payin", "REMARKS": "Payin Above 30%"},
#           {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "TATA", "PO": "90% of Payin", "REMARKS": "Payin Above 40%"},
#             {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "TATA", "PO": "90% of Payin", "REMARKS": "Payin Above 50%"},



  

#     {"LOB": "CV", "SEGMENT": "All GVW & PCV 3W, GCV 3W", "INSURER": "TATA", "PO": "-2%", "REMARKS": "Payin Below 20%"},
#     {"LOB": "CV", "SEGMENT": "All GVW & PCV 3W, GCV 3W", "INSURER": "TATA", "PO": "-3%", "REMARKS": "Payin 21% to 30%"},
#     {"LOB": "CV", "SEGMENT": "All GVW & PCV 3W, GCV 3W", "INSURER": "TATA", "PO": "-4%", "REMARKS": "Payin 31% to 50%"},
#     {"LOB": "CV", "SEGMENT": "All GVW & PCV 3W, GCV 3W", "INSURER": "TATA", "PO": "-5%", "REMARKS": "Payin Above 50%"},


#     {"LOB": "BUS", "SEGMENT": "SCHOOL BUS", "INSURER": "TATA", "PO": "Less 2% of Payin", "REMARKS": "NIL"},

#     {"LOB": "BUS", "SEGMENT": "STAFF BUS", "INSURER": "TATA", "PO": "88% of Payin", "REMARKS": "NIL"},


#     {"LOB": "TAXI", "SEGMENT": "TAXI", "INSURER": "TATA", "PO": "-2%", "REMARKS": "Payin Below 20%"},
#     {"LOB": "TAXI", "SEGMENT": "TAXI", "INSURER": "TATA", "PO": "-3%", "REMARKS": "Payin 21% to 30%"},
#     {"LOB": "TAXI", "SEGMENT": "TAXI", "INSURER": "TATA", "PO": "-4%", "REMARKS": "Payin 31% to 50%"},
#     {"LOB": "TAXI", "SEGMENT": "TAXI", "INSURER": "TATA", "PO": "-5%", "REMARKS": "Payin Above 50%"},



#     {"LOB": "MISD", "SEGMENT": "Misd, Tractor", "INSURER": "TATA", "PO": "88% of Payin", "REMARKS": "NIL"}
# ]

# def extract_text_from_file(file_bytes: bytes, filename: str, content_type: str) -> str:
#     """Extract text from uploaded image file using GPT-4o"""
#     file_extension = filename.split('.')[-1].lower() if '.' in filename else ''
    
#     if file_extension not in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'] and not content_type.startswith('image/'):
#         raise ValueError(f"Unsupported file type: {filename}")
    
#     try:
#         image_base64 = base64.b64encode(file_bytes).decode('utf-8')
        
#         prompt = """
# You are extracting insurance policy data from an image. Return a JSON array with these exact keys: segment, policy_type, location, payin, remark.

# STEP-BY-STEP EXTRACTION:

# STEP 1: Identify the vehicle/policy category
# - 2W, MC, MCY, SC, Scooter, EV → TWO WHEELER
# - PVT CAR, Car, PCI → PRIVATE CAR  
# - CV, GVW, PCV, GCV, tonnage → COMMERCIAL VEHICLE
# - Bus → BUS
# - Taxi → TAXI
# - Tractor, Ambulance, Misd → MISCELLANEOUS

# STEP 2: Identify policy type from columns
# - 1+1 column = Comp
# - SATP column = TP
# - If both exist, create TWO separate records

# STEP 3: Map to EXACT segment (MANDATORY):

# TWO WHEELER:
#   IF 1+1 OR Comp OR SAOD → segment = "TW SAOD + COMP"
#   IF SATP OR TP → segment = "TW TP"
#   IF New/Fresh/1+5 → segment = "1+5"
#   NEVER use "2W", "MC", "Scooter" as segment

# PRIVATE CAR:
#   IF 1+1 OR Comp OR SAOD → segment = "PVT CAR COMP + SAOD"
#   IF SATP OR TP → segment = "PVT CAR TP"
#   and 4W means 4 wheeler means Private Car 

# COMMERCIAL VEHICLE:
#   ALWAYS → segment = "All GVW & PCV 3W, GCV 3W"
#   (Digit treats all CV the same regardless of tonnage)

# BUS:
#   IF School → segment = "SCHOOL BUS"
#   ELSE → segment = "STAFF BUS"

# TAXI:
#   segment = "TAXI"

# MISCELLANEOUS:
#   segment = "Misd, Tractor"

# STEP 4: Extract other fields
# - policy_type: "Comp" or "TP"
# - location: Cluster/Agency name
# - payin: ONLY CD2 value as NUMBER (ignore CD1)
# - remark: Additional details as STRING

# CRITICAL RULES:
# - payin must be numeric (63.0 not "63.0%")
# - Create separate records if both 1+1 and SATP columns exist
# - NEVER use raw names like "2W" in segment
# - Handle negative % as positive

# --- TATA-AIG CV GRID OVERRIDE (APPLY ONLY IF "TATA" OR "Forwarded" IS VISIBLE) ---
# IF the image contains the word **TATA**, **Forwarded**, or a grid titled "Grid Applicable for June 25":


# 2. Extract **EVERY ROW** in the table as a **separate JSON record**.
# 3. For **segment** use the exact tonnage/type shown  it comes under segment "All GVW & PCV 3W, GCV 3W". 
#    - "2.5–3.5 T"             → "All GVW & PCV 3W, GCV 3W"
#    - "<2.5 T"                →"All GVW & PCV 3W, GCV 3W"
#    - "3.5–12 T"              →"All GVW & PCV 3W, GCV 3W"
#    - "PCV 3W"                → "All GVW & PCV 3W, GCV 3W"
#    - ">3.5-45 T SATP"        → "All GVW & PCV 3W, GCV 3W"
#    - "12-45 T (> 6 years)"   →"All GVW & PCV 3W, GCV 3W"
#    - Any other tonnage line  → "All GVW & PCV 3W, GCV 3W"
# 4. **policy_type** = "TP" for all rows (the table has no Comp column).
# 5. **location** = list of cities in that row (e.g., "Mumbai, Pune" or "Mumbai, Pune, Nagpur").
# 6. **payin** = the number before the "x" (e.g., "68x" → 68, "51x" → 51). Remove the "x", return plain integer.
# 7. **remark** = the full tonnage text from the row (e.g., "<2.5 T", ">3.5-45 T SATP").

# Do **not** collapse rows into a single record.  
# If the image is **not** a TATA grid, fall back to the original rules above.
# --- END OF OVERRIDE ---

# Return ONLY JSON array, no markdown.
# """
       
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[{
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt},
#                     {"type": "image_url", "image_url": {"url": f"data:image/{file_extension};base64,{image_base64}"}}
#                 ]
#             }],
#             temperature=0.0,
#             max_tokens=4000
#         )
        
#         extracted_text = response.choices[0].message.content.strip()
        
#         # Clean markdown formatting
#         cleaned_text = re.sub(r'```json\s*|\s*```', '', extracted_text).strip()
        
#         # Extract JSON array
#         start_idx = cleaned_text.find('[')
#         end_idx = cleaned_text.rfind(']') + 1
#         if start_idx != -1 and end_idx > start_idx:
#             cleaned_text = cleaned_text[start_idx:end_idx]
        
#         # Validate JSON
#         json.loads(cleaned_text)
#         return cleaned_text
        
#     except Exception as e:
#         logger.error(f"Error in OCR extraction: {str(e)}")
#         return "[]"

# def classify_payin(payin_value):
#     """Classify payin into categories"""
#     try:
#         if isinstance(payin_value, (int, float)):
#             payin_float = float(payin_value)
#         else:
#             payin_clean = str(payin_value).replace('%', '').replace(' ', '').replace('-', '').strip()
#             if not payin_clean or payin_clean.upper() == 'N/A':
#                 return 0.0, "Payin Below 20%"
#             payin_float = float(payin_clean)
        
#         if payin_float <= 20:
#             return payin_float, "Payin Below 20%"
#         elif payin_float <= 30:
#             return payin_float, "Payin 21% to 30%"
#         elif payin_float <= 50:
#             return payin_float, "Payin 31% to 50%"
#         else:
#             return payin_float, "Payin Above 50%"
#     except (ValueError, TypeError) as e:
#         logger.warning(f"Could not parse payin: {payin_value}, error: {e}")
#         return 0.0, "Payin Below 20%"

# def determine_lob(segment: str) -> str:
#     """Determine LOB from segment"""
#     segment_upper = segment.upper()
    
#     if any(kw in segment_upper for kw in ['TW', '2W', 'MC', 'SC', '1+5']):
#         return "TW"
#     elif any(kw in segment_upper for kw in ['PVT CAR', 'CAR', 'PCI']):
#         return "PVT CAR"
#     elif any(kw in segment_upper for kw in ['CV', 'GVW', 'PCV', 'GCV']):
#         return "CV"
#     elif 'BUS' in segment_upper:
#         return "BUS"
#     elif 'TAXI' in segment_upper:
#         return "TAXI"
#     elif any(kw in segment_upper for kw in ['MISD', 'TRACTOR']):
#         return "MISD"
    
#     return "UNKNOWN"

# def apply_formula(policy_data):
#     """Apply formula rules and calculate payouts"""
#     if not policy_data:
#         return []
    
#     calculated_data = []
    
#     for record in policy_data:
#         try:
#             segment = str(record.get('segment', ''))
#             payin_value = record.get('Payin_Value', 0)
#             payin_category = record.get('Payin_Category', '')
            
#             lob = determine_lob(segment)
#             segment_upper = segment.upper()
            
#             # Find matching rule
#             matched_rule = None
#             for rule in FORMULA_DATA:
#                 # Match LOB
#                 if rule["LOB"] != lob:
#                     continue
                
#                 # Match Segment
#                 rule_segment = rule["SEGMENT"].upper()
#                 if rule_segment not in segment_upper:
#                     continue
                
#                 # Match Payin Category or NIL
#                 remarks = rule.get("REMARKS", "")
#                 if remarks == "NIL" or payin_category in remarks:
#                     matched_rule = rule
#                     break
            
#             # Calculate payout
#             if matched_rule:
#                 po_formula = matched_rule["PO"]
#                 calculated_payout = payin_value
                
#                 if "90% of Payin" in po_formula:
#                     calculated_payout *= 0.9
#                 elif "88% of Payin" in po_formula:
#                     calculated_payout *= 0.88
#                 elif "Less 2%" in po_formula or "-2%" in po_formula:
#                     calculated_payout -= 2
#                 elif "-3%" in po_formula:
#                     calculated_payout -= 3
#                 elif "-4%" in po_formula:
#                     calculated_payout -= 4
#                 elif "-5%" in po_formula:
#                     calculated_payout -= 5
                
#                 calculated_payout = max(0, calculated_payout)
#                 formula_used = po_formula
#                 rule_explanation = f"Match: LOB={lob}, Segment={rule_segment}, {remarks}"
#             else:
#                 calculated_payout = payin_value
#                 formula_used = "No matching rule"
#                 rule_explanation = f"No rule for LOB={lob}, Segment={segment_upper}"
            
#             # Format remark
#             remark_value = record.get('remark', '')
#             if isinstance(remark_value, list):
#                 remark_value = '; '.join(str(r) for r in remark_value)
            
#             calculated_data.append({
#                 'segment': segment,
#                 'policy type': record.get('policy_type', 'Comp'),
#                 'location': record.get('location', 'N/A'),
#                 'payin': f"{payin_value:.2f}%",
#                 'remark': str(remark_value),
#                 'Calculated Payout': f"{calculated_payout:.2f}%",
#                 'Formula Used': formula_used,
#                 'Rule Explanation': rule_explanation
#             })
            
#         except Exception as e:
#             logger.error(f"Error processing record {record}: {str(e)}")
#             calculated_data.append({
#                 'segment': str(record.get('segment', 'Unknown')),
#                 'policy type': record.get('policy_type', 'Comp'),
#                 'location': record.get('location', 'N/A'),
#                 'payin': str(record.get('payin', '0%')),
#                 'remark': str(record.get('remark', 'Error')),
#                 'Calculated Payout': "Error",
#                 'Formula Used': "Error",
#                 'Rule Explanation': f"Error: {str(e)}"
#             })
    
#     return calculated_data

# def process_files(policy_file_bytes: bytes, policy_filename: str, policy_content_type: str, company_name: str):
#     """Main processing function"""
#     try:
#         logger.info(f"🚀 Processing {policy_filename} for {company_name}")
        
#         # Extract text
#         extracted_text = extract_text_from_file(policy_file_bytes, policy_filename, policy_content_type)
        
#         if not extracted_text or extracted_text == "[]":
#             raise ValueError("No text extracted from image")
        
#         # Parse JSON
#         policy_data = json.loads(extracted_text)
#         if isinstance(policy_data, dict):
#             policy_data = [policy_data]
        
#         if not policy_data:
#             raise ValueError("No policy data found")
        
#         logger.info(f"✅ Parsed {len(policy_data)} records")
        
#         # Classify payin
#         for record in policy_data:
#             payin_val, payin_cat = classify_payin(record.get('payin', 0))
#             record['Payin_Value'] = payin_val
#             record['Payin_Category'] = payin_cat
        
#         # Apply formulas
#         calculated_data = apply_formula(policy_data)
        
#         if not calculated_data:
#             raise ValueError("No data after formula application")
        
#         logger.info(f"✅ Calculated {len(calculated_data)} records")
        
#         # Create Excel
#         df = pd.DataFrame(calculated_data)
#         output = BytesIO()
        
#         with pd.ExcelWriter(output, engine='openpyxl') as writer:
#             df.to_excel(writer, sheet_name='Policy Data', startrow=2, index=False)
#             worksheet = writer.sheets['Policy Data']
            
#             # Format headers
#             for col_num, value in enumerate(df.columns, 1):
#                 cell = worksheet.cell(row=3, column=col_num, value=value)
#                 cell.font = cell.font.copy(bold=True)
            
#             # Add title
#             title_cell = worksheet.cell(row=1, column=1, value=f"{company_name} - Policy Data")
#             worksheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(df.columns))
#             title_cell.font = title_cell.font.copy(bold=True, size=14)
#             title_cell.alignment = title_cell.alignment.copy(horizontal='center')
        
#         output.seek(0)
#         excel_data_base64 = base64.b64encode(output.read()).decode('utf-8')
        
#         # Calculate metrics
#         avg_payin = sum([r['Payin_Value'] for r in policy_data]) / len(policy_data)
#         formula_summary = {}
#         for record in calculated_data:
#             formula = record['Formula Used']
#             formula_summary[formula] = formula_summary.get(formula, 0) + 1
        
#         return {
#             "extracted_text": extracted_text,
#             "parsed_data": policy_data,
#             "calculated_data": calculated_data,
#             "excel_data": excel_data_base64,
#             "csv_data": df.to_csv(index=False),
#             "json_data": json.dumps(calculated_data, indent=2),
#             "formula_data": FORMULA_DATA,
#             "metrics": {
#                 "total_records": len(calculated_data),
#                 "avg_payin": round(avg_payin, 1),
#                 "unique_segments": len(set([r['segment'] for r in calculated_data])),
#                 "company_name": company_name,
#                 "formula_summary": formula_summary
#             }
#         }
    
#     except Exception as e:
#         logger.error(f"Error in process_files: {str(e)}", exc_info=True)
#         raise

# @app.get("/", response_class=HTMLResponse)
# async def root():
#     """Serve HTML frontend"""
#     html_path = Path("index.html")
#     if html_path.exists():
#         return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
#     return HTMLResponse(content="<h1>Insurance Policy Processing System</h1><p>Upload via POST /process</p>")

# @app.post("/process")
# async def process_policy(company_name: str = Form(...), policy_file: UploadFile = File(...)):
#     """Process policy image"""
#     try:
#         policy_file_bytes = await policy_file.read()
#         if not policy_file_bytes:
#             return JSONResponse(status_code=400, content={"error": "Empty file"})
        
#         results = process_files(policy_file_bytes, policy_file.filename, policy_file.content_type, company_name)
#         return JSONResponse(content=results)
        
#     except ValueError as e:
#         return JSONResponse(status_code=400, content={"error": str(e)})
#     except Exception as e:
#         logger.error(f"Error: {str(e)}", exc_info=True)
#         return JSONResponse(status_code=500, content={"error": f"Processing failed: {str(e)}"})

# @app.get("/health")
# async def health_check():
#     """Health check"""
#     return JSONResponse(content={"status": "healthy"})

# if __name__ == "__main__":
#     import uvicorn
#     logger.info("🚀 Starting server at http://localhost:8000")
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import base64
import json
import os
from dotenv import load_dotenv
import logging
import re
import pandas as pd
from openai import OpenAI
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Load OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("Warning: OPENAI_API_KEY environment variable not set")
    raise RuntimeError("OPENAI_API_KEY environment variable not set")

# Initialize OpenAI client
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    raise RuntimeError(f"Failed to initialize OpenAI client: {str(e)}")

app = FastAPI(title="Insurance Policy Processing System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simplified Formula Data - Only for ICICI
FORMULA_DATA = [
    {"LOB": "TW", "SEGMENT": "1+5", "INSURER": "TATA", "PO": "90% of Payin", "REMARKS": "NIL"},
    {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "INSURER": "TATA", "PO": "90% of Payin", "REMARKS": "NIL"},
    
    {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "TATA", "PO": "-2%", "REMARKS": "Payin Below 20%"},
    {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "TATA", "PO": "-3%", "REMARKS": "Payin 21% to 30%"},
    {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "TATA", "PO": "-4%", "REMARKS": "Payin 31% to 50%"},
    {"LOB": "TW", "SEGMENT": "TW TP", "INSURER": "TATA", "PO": "-5%", "REMARKS": "Payin Above 50%"},

    {"LOB": "PVT CAR", "SEGMENT": "PVT CAR COMP + SAOD", "INSURER": "TATA", "PO": "90% of Payin", "REMARKS": "Payin Below 20%"},
    {"LOB": "PVT CAR", "SEGMENT": "PVT CAR COMP + SAOD", "INSURER": "TATA", "PO": "90% of Payin", "REMARKS": "Payin Above 20%"},
    {"LOB": "PVT CAR", "SEGMENT": "PVT CAR COMP + SAOD", "INSURER": "TATA", "PO": "90% of Payin", "REMARKS": "Payin Above 30%"},
    {"LOB": "PVT CAR", "SEGMENT": "PVT CAR COMP + SAOD", "INSURER": "TATA", "PO": "90% of Payin", "REMARKS": "Payin Above 40%"},
    {"LOB": "PVT CAR", "SEGMENT": "PVT CAR COMP + SAOD", "INSURER": "TATA", "PO": "90% of Payin", "REMARKS": "Payin Above 50%"},

    {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "TATA", "PO": "90% of Payin", "REMARKS": "Payin Below 20%"},
    {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "TATA", "PO": "90% of Payin", "REMARKS": "Payin Above 20%"},
    {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "TATA", "PO": "90% of Payin", "REMARKS": "Payin Above 30%"},
    {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "TATA", "PO": "90% of Payin", "REMARKS": "Payin Above 40%"},
    {"LOB": "PVT CAR", "SEGMENT": "PVT CAR TP", "INSURER": "TATA", "PO": "90% of Payin", "REMARKS": "Payin Above 50%"},

    {"LOB": "CV", "SEGMENT": "All GVW & PCV 3W, GCV 3W", "INSURER": "TATA", "PO": "-2%", "REMARKS": "Payin Below 20%"},
    {"LOB": "CV", "SEGMENT": "All GVW & PCV 3W, GCV 3W", "INSURER": "TATA", "PO": "-3%", "REMARKS": "Payin 21% to 30%"},
    {"LOB": "CV", "SEGMENT": "All GVW & PCV 3W, GCV 3W", "INSURER": "TATA", "PO": "-4%", "REMARKS": "Payin 31% to 50%"},
    {"LOB": "CV", "SEGMENT": "All GVW & PCV 3W, GCV 3W", "INSURER": "TATA", "PO": "-5%", "REMARKS": "Payin Above 50%"},

    {"LOB": "BUS", "SEGMENT": "SCHOOL BUS", "INSURER": "TATA", "PO": "Less 2% of Payin", "REMARKS": "NIL"},
    {"LOB": "BUS", "SEGMENT": "STAFF BUS", "INSURER": "TATA", "PO": "88% of Payin", "REMARKS": "NIL"},

    {"LOB": "TAXI", "SEGMENT": "TAXI", "INSURER": "TATA", "PO": "-2%", "REMARKS": "Payin Below 20%"},
    {"LOB": "TAXI", "SEGMENT": "TAXI", "INSURER": "TATA", "PO": "-3%", "REMARKS": "Payin 21% to 30%"},
    {"LOB": "TAXI", "SEGMENT": "TAXI", "INSURER": "TATA", "PO": "-4%", "REMARKS": "Payin 31% to 50%"},
    {"LOB": "TAXI", "SEGMENT": "TAXI", "INSURER": "TATA", "PO": "-5%", "REMARKS": "Payin Above 50%"},

    {"LOB": "MISD", "SEGMENT": "Misd, Tractor", "INSURER": "TATA", "PO": "88% of Payin", "REMARKS": "NIL"}
]

def extract_text_from_file(file_bytes: bytes, filename: str, content_type: str) -> str:
    """Extract text from uploaded image file using GPT-4o"""
    file_extension = filename.split('.')[-1].lower() if '.' in filename else ''
    
    if file_extension not in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'] and not content_type.startswith('image/'):
        raise ValueError(f"Unsupported file type: {filename}")
    
    try:
        image_base64 = base64.b64encode(file_bytes).decode('utf-8')
        
#         prompt = """
# You are extracting insurance policy data from an image. Return a JSON array with these exact keys: segment, policy_type, location, payin, remark.

# STEP-BY-STEP EXTRACTION:

# STEP 1: Identify the vehicle/policy category
# - 2W, MC, MCY, SC, Scooter, EV → TWO WHEELER
# - PVT CAR, Car, PCI → PRIVATE CAR  
# - CV, GVW, PCV, GCV, tonnage → COMMERCIAL VEHICLE
# - Bus → BUS
# - Taxi → TAXI
# - Tractor, Ambulance, Misd → MISCELLANEOUS

# STEP 2: Identify policy type from columns
# - 1+1 column = Comp
# - SATP column = TP
# - If both exist, create TWO separate records

# STEP 3: Map to EXACT segment (MANDATORY):

# TWO WHEELER:
#   IF 1+1 OR Comp OR SAOD → segment = "TW SAOD + COMP"
#   IF SATP OR TP → segment = "TW TP"
#   IF New/Fresh/1+5 → segment = "1+5"
#   NEVER use "2W", "MC", "Scooter" as segment

# PRIVATE CAR:
#   IF 1+1 OR Comp OR SAOD → segment = "PVT CAR COMP + SAOD"
#   IF SATP OR TP → segment = "PVT CAR TP"
#   and 4W means 4 wheeler means Private Car 

# COMMERCIAL VEHICLE:
#   ALWAYS → segment = "All GVW & PCV 3W, GCV 3W"
#   (Digit treats all CV the same regardless of tonnage)

# BUS:
#   IF School → segment = "SCHOOL BUS"
#   ELSE → segment = "STAFF BUS"

# TAXI:
#   segment = "TAXI"

# MISCELLANEOUS:
#   segment = "Misd, Tractor"

# STEP 4: Extract other fields
# - policy_type: "Comp" or "TP"
# - location: Cluster/Agency name
# - payin: ONLY CD2 value as NUMBER (ignore CD1)
# - remark: Additional details as STRING

# CRITICAL RULES:
# - payin must be numeric (63.0 not "63.0%")
# - Create separate records if both 1+1 and SATP columns exist
# - NEVER use raw names like "2W" in segment
# - Handle negative % as positive

# --- TATA-AIG CV GRID OVERRIDE (APPLY ONLY IF "TATA" OR "Forwarded" IS VISIBLE) ---
# IF the image contains the word **TATA**, **Forwarded**, or a grid titled "Grid Applicable for June 25":


# 2. Extract **EVERY ROW** in the table as a **separate JSON record**.
# 3. For **segment** use the exact tonnage/type shown  it comes under segment "All GVW & PCV 3W, GCV 3W". 
#    - "2.5–3.5 T"             → "All GVW & PCV 3W, GCV 3W"
#    - "<2.5 T"                →"All GVW & PCV 3W, GCV 3W"
#    - "3.5–12 T"              →"All GVW & PCV 3W, GCV 3W"
#    - "PCV 3W"                → "All GVW & PCV 3W, GCV 3W"
#    - ">3.5-45 T SATP"        → "All GVW & PCV 3W, GCV 3W"
#    - "12-45 T (> 6 years)"   →"All GVW & PCV 3W, GCV 3W"
#    - Any other tonnage line  → "All GVW & PCV 3W, GCV 3W"
# 4. **policy_type** = "TP" for all rows (the table has no Comp column).
# 5. **location** = list of cities in that row (e.g., "Mumbai, Pune" or "Mumbai, Pune, Nagpur").
# 6. **payin** = the number before the "x" (e.g., "68x" → 68, "51x" → 51). Remove the "x", return plain integer.
# 7. **remark** = the full tonnage text from the row (e.g., "<2.5 T", ">3.5-45 T SATP").

# Do **not** collapse rows into a single record.  
# If the image is **not** a TATA grid, fall back to the original rules above.
# --- END OF OVERRIDE ---

# Return ONLY JSON array, no markdown.
# """
        prompt = """
You are extracting insurance policy data from an image. Return a JSON array with these exact keys: segment, policy_type, location, payin, remark.

STEP-BY-STEP EXTRACTION:

STEP 1: Identify the vehicle/policy category
- 2W, MC, MCY, SC, Scooter, EV → TWO WHEELER
- PVT CAR, Car, PCI → PRIVATE CAR  
- CV, GVW, PCV, GCV, tonnage → COMMERCIAL VEHICLE
- Bus → BUS
- Taxi → TAXI
- Tractor, Ambulance, Misd → MISCELLANEOUS

STEP 2: Identify policy type from columns
- 1+1 column = Comp
- SATP column = TP
- If both exist, create TWO separate records

STEP 3: Map to EXACT segment (MANDATORY):

TWO WHEELER:
  IF 1+1 OR Comp OR SAOD → segment = "TW SAOD + COMP"
  IF SATP OR TP → segment = "TW TP"
  IF New/Fresh/1+5 → segment = "1+5"
  NEVER use "2W", "MC", "Scooter" as segment

PRIVATE CAR:
  IF 1+1 OR Comp OR SAOD → segment = "PVT CAR COMP + SAOD"
  IF SATP OR TP → segment = "PVT CAR TP"
  and 4W means 4 wheeler means Private Car 

COMMERCIAL VEHICLE:
  ALWAYS → segment = "All GVW & PCV 3W, GCV 3W"
  (Digit treats all CV the same regardless of tonnage)

BUS:
  IF School → segment = "SCHOOL BUS"
  ELSE → segment = "STAFF BUS"

TAXI:
  segment = "TAXI"

MISCELLANEOUS:
  segment = "Misd, Tractor"

STEP 4: Extract other fields
- policy_type: "Comp" or "TP"
- location: Cluster/Agency name
- payin: ONLY CD2 value as NUMBER (ignore CD1)
- remark: Additional details as STRING

CRITICAL RULES:
- payin must be numeric (63.0 not "63.0%")
- Create separate records if both 1+1 and SATP columns exist
- NEVER use raw names like "2W" in segment
- Handle negative % as positive

--- TATA-AIG CV GRID OVERRIDE (APPLY ONLY IF "TATA" OR "Forwarded" IS VISIBLE) ---
IF the image contains the word **TATA**, **Forwarded**, or a grid titled "Grid Applicable for June 25":


2. Extract **EVERY ROW** in the table as a **separate JSON record**.
3. For **segment** use the exact tonnage/type shown  it comes under segment "All GVW & PCV 3W, GCV 3W". 
   - "2.5–3.5 T"             → "All GVW & PCV 3W, GCV 3W"
   - "<2.5 T"                →"All GVW & PCV 3W, GCV 3W"
   - "3.5–12 T"              →"All GVW & PCV 3W, GCV 3W"
   - "PCV 3W"                → "All GVW & PCV 3W, GCV 3W"
   - ">3.5-45 T SATP"        → "All GVW & PCV 3W, GCV 3W"
   - "12-45 T (> 6 years)"   →"All GVW & PCV 3W, GCV 3W"
   - Any other tonnage line  → "All GVW & PCV 3W, GCV 3W"
4. **policy_type** = "TP" for all rows (the table has no Comp column).
5. **location** = list of cities in that row (e.g., "Mumbai, Pune" or "Mumbai, Pune, Nagpur").
6. **payin** = the number before the "x" (e.g., "68x" → 68, "51x" → 51). Remove the "x", return plain integer.
7. **remark** = the full tonnage text from the row (e.g., "<2.5 T", ">3.5-45 T SATP").

Do **not** collapse rows into a single record.  
If the image is **not** a TATA grid, fall back to the original rules above.
--- END OF OVERRIDE ---

**IMPORTANT: If there are separate columns for Petrol, Diesel, CNG, or any other fuel type, extract each as a separate record with the same segment, policy_type, and location, but include the fuel type and its corresponding payin value. For example, if a row has Petrol 55% and Diesel 45%, create two records: one with payin=55 and remark including "Petrol", another with payin=45 and remark including "Diesel".**

Let me train you on one data

So, here is the data 

+---------------------------+---------------------------+---------------------------+---------------------------+---------------------------+
| School bus > 11 SC        |                           | Pan India except ROTN, MP 1,2 &3                         |                           |
+---------------------------+---------------------------+---------------------------+---------------------------+---------------------------+
|                           | Additional- Yearly /      |                           |                           |                           |
| YTD Target ( in Cr)       | qtry                      | Institute                 | Individual                | TATA < 11 SC on all RTO's |
+---------------------------+---------------------------+---------------------------+---------------------------+---------------------------+
| >35                       | 2.5%                      | 75.00%                    | 72.00%                    | 50%                       |
+---------------------------+---------------------------+---------------------------+---------------------------+---------------------------+
|                           |                           | * ROTN- 65 %,             |                           |                           |
|                           |                           | MP 1, MP 2 & MP 3 - 67%   |                           |                           |
|                           |                           | <11 Seating 50% max       |                           |                           |
+---------------------------+---------------------------+---------------------------+---------------------------+---------------------------+

the interpretation is as follows:

School Bus segment and in remark write it is 11 seater
all india rate for individual is 72%
and for institute it is 75%

tata <11 seater all india is 50

except for the ROTN,MP-1,2 & 3

For ROTN -> School Bus >11 Seater is 65%
also for the MP-1,2 &3 it is 67%

for <11 seater it is 50%

so this was it , this was the interpretation




Return ONLY JSON array, no markdown.
"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/{file_extension};base64,{image_base64}"}}
                ]
            }],
            temperature=0.0,
            max_tokens=4000
        )
        
        extracted_text = response.choices[0].message.content.strip()
        
        # Clean markdown formatting
        cleaned_text = re.sub(r'```json\s*|\s*```', '', extracted_text).strip()
        
        # Extract JSON array
        start_idx = cleaned_text.find('[')
        end_idx = cleaned_text.rfind(']') + 1
        if start_idx != -1 and end_idx > start_idx:
            cleaned_text = cleaned_text[start_idx:end_idx]
        
        # Validate JSON
        json.loads(cleaned_text)
        return cleaned_text
        
    except Exception as e:
        logger.error(f"Error in OCR extraction: {str(e)}")
        return "[]"

def determine_lob(segment: str) -> str:
    """Determine LOB from segment"""
    segment_upper = segment.upper()
    
    if 'BUS' in segment_upper:
        return "BUS"
    elif any(kw in segment_upper for kw in ['TW', '2W', 'MC', 'SC', '1+5']):
        return "TW"
    elif any(kw in segment_upper for kw in ['PVT CAR', 'CAR', 'PCI']):
        return "PVT CAR"
    elif any(kw in segment_upper for kw in ['CV', 'GVW', 'PCV', 'GCV']):
        return "CV"
    elif 'TAXI' in segment_upper:
        return "TAXI"
    elif any(kw in segment_upper for kw in ['MISD', 'TRACTOR']):
        return "MISD"
    
    return "UNKNOWN"

def apply_formula(policy_data):
    """Apply formula rules and calculate payouts - SEGMENT-AWARE"""
    if not policy_data:
        return []
    
    calculated_data = []
    
    for record in policy_data:
        try:
            segment = str(record.get('segment', '')).strip()
            payin_raw = record.get('payin', 0)
            policy_type = record.get('policy_type', 'Comp')
            location = record.get('location', 'N/A')
            remark = record.get('remark', '')
            if isinstance(remark, list):
                remark = '; '.join(str(r) for r in remark)
            remark = str(remark)

            # Parse payin to float
            try:
                payin_value = float(str(payin_raw).replace('%', '').replace('x', '').strip())
                payin_value = max(0, payin_value)
            except:
                payin_value = 0.0

            lob = determine_lob(segment)
            segment_upper = segment.upper()

            # Find all rules for this LOB and Segment
            matching_rules = []
            for rule in FORMULA_DATA:
                if rule["LOB"] != lob:
                    continue
                if rule["SEGMENT"].upper() not in segment_upper:
                    continue
                matching_rules.append(rule)

            if not matching_rules:
                calculated_data.append({
                    'segment': segment,
                    'policy type': policy_type,
                    'location': location,
                    'payin': f"{payin_value:.2f}%",
                    'remark': remark,
                    'Calculated Payout': f"{payin_value:.2f}%",
                    'Formula Used': "No matching rule",
                    'Rule Explanation': f"No rule for LOB={lob}, Segment={segment}"
                })
                continue

            # Helper: get threshold for sorting
            def get_threshold(remark):
                remark = remark.strip()
                if remark == "NIL":
                    return float('inf')
                if "Below 20" in remark:
                    return 20
                if "21% to 30" in remark:
                    return 30
                if "31% to 50" in remark:
                    return 50
                if "Above 50" in remark:
                    return float('inf')
                if "Above 40" in remark:
                    return 40
                if "Above 30" in remark:
                    return 30
                if "Above 20" in remark:
                    return 20
                return 0

            # Sort rules: lower threshold first, NIL last
            sorted_rules = sorted(
                matching_rules,
                key=lambda r: (get_threshold(r["REMARKS"]), 0 if r["REMARKS"] == "NIL" else 1)
            )

            # Find matching rule
            applied_rule = None
            explanation = ""

            for rule in sorted_rules:
                remarks = rule["REMARKS"]
                if remarks == "NIL":
                    applied_rule = rule
                    explanation = "NIL condition"
                    break

                if "Below 20" in remarks and payin_value < 20:
                    applied_rule = rule
                    explanation = f"payin {payin_value} < 20"
                    break
                elif "21% to 30" in remarks and 21 <= payin_value <= 30:
                    applied_rule = rule
                    explanation = f"payin {payin_value} in [21,30]"
                    break
                elif "31% to 50" in remarks and 31 <= payin_value <= 50:
                    applied_rule = rule
                    explanation = f"payin {payin_value} in [31,50]"
                    break
                elif "Above 50" in remarks and payin_value > 50:
                    applied_rule = rule
                    explanation = f"payin {payin_value} > 50"
                    break
                elif "Above 40" in remarks and payin_value > 40:
                    applied_rule = rule
                    explanation = f"payin {payin_value} > 40"
                    break
                elif "Above 30" in remarks and payin_value > 30:
                    applied_rule = rule
                    explanation = f"payin {payin_value} > 30"
                    break
                elif "Above 20" in remarks and payin_value > 20:
                    applied_rule = rule
                    explanation = f"payin {payin_value} > 20"
                    break

            # Fallback
            if not applied_rule:
                applied_rule = matching_rules[0]
                explanation = "fallback"

            # Apply formula
            po_formula = applied_rule["PO"]
            payout = payin_value

            if "90% of Payin" in po_formula:
                payout *= 0.90
            elif "88% of Payin" in po_formula:
                payout *= 0.88
            elif "Less 2% of Payin" in po_formula:
                payout = max(0, payout - 2)
            elif "-2%" in po_formula:
                payout = max(0, payout - 2)
            elif "-3%" in po_formula:
                payout = max(0, payout - 3)
            elif "-4%" in po_formula:
                payout = max(0, payout - 4)
            elif "-5%" in po_formula:
                payout = max(0, payout - 5)

            rule_explanation = f"Match: LOB={lob}, Segment={applied_rule['SEGMENT']}, {explanation}"

            calculated_data.append({
                'segment': segment,
                'policy type': policy_type,
                'location': location,
                'payin': f"{payin_value:.2f}%",
                'remark': remark,
                'Calculated Payout': f"{payout:.2f}%",
                'Formula Used': po_formula,
                'Rule Explanation': rule_explanation
            })

        except Exception as e:
            logger.error(f"Error processing record {record}: {str(e)}")
            calculated_data.append({
                'segment': str(record.get('segment', 'Error')),
                'policy type': 'Error',
                'location': 'Error',
                'payin': 'Error',
                'remark': 'Error',
                'Calculated Payout': "Error",
                'Formula Used': "Error",
                'Rule Explanation': f"Error: {str(e)}"
            })

    return calculated_data

def process_files(policy_file_bytes: bytes, policy_filename: str, policy_content_type: str, company_name: str):
    """Main processing function"""
    try:
        logger.info(f"Processing {policy_filename} for {company_name}")
        
        # Extract text
        extracted_text = extract_text_from_file(policy_file_bytes, policy_filename, policy_content_type)
        
        if not extracted_text or extracted_text == "[]":
            raise ValueError("No text extracted from image")
        
        # Parse JSON
        policy_data = json.loads(extracted_text)
        if isinstance(policy_data, dict):
            policy_data = [policy_data]
        
        if not policy_data:
            raise ValueError("No policy data found")
        
        logger.info(f"Parsed {len(policy_data)} records")
        
        # Apply formulas (classification now inside)
        calculated_data = apply_formula(policy_data)
        
        if not calculated_data:
            raise ValueError("No data after formula application")
        
        logger.info(f"Calculated {len(calculated_data)} records")
        
        # Create Excel
        df = pd.DataFrame(calculated_data)
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Policy Data', startrow=2, index=False)
            worksheet = writer.sheets['Policy Data']
            
            # Format headers
            for col_num, value in enumerate(df.columns, 1):
                cell = worksheet.cell(row=3, column=col_num, value=value)
                cell.font = cell.font.copy(bold=True)
            
            # Add title
            title_cell = worksheet.cell(row=1, column=1, value=f"{company_name} - Policy Data")
            worksheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(df.columns))
            title_cell.font = title_cell.font.copy(bold=True, size=14)
            title_cell.alignment = title_cell.alignment.copy(horizontal='center')
        
        output.seek(0)
        excel_data_base64 = base64.b64encode(output.read()).decode('utf-8')
        
        # Calculate metrics
        avg_payin = sum([float(r['payin'].strip('%')) for r in calculated_data if r['payin'] != 'Error']) / len(calculated_data) if calculated_data else 0
        formula_summary = {}
        for record in calculated_data:
            formula = record['Formula Used']
            formula_summary[formula] = formula_summary.get(formula, 0) + 1
        
        return {
            "extracted_text": extracted_text,
            "parsed_data": policy_data,
            "calculated_data": calculated_data,
            "excel_data": excel_data_base64,
            "csv_data": df.to_csv(index=False),
            "json_data": json.dumps(calculated_data, indent=2),
            "formula_data": FORMULA_DATA,
            "metrics": {
                "total_records": len(calculated_data),
                "avg_payin": round(avg_payin, 1),
                "unique_segments": len(set([r['segment'] for r in calculated_data])),
                "company_name": company_name,
                "formula_summary": formula_summary
            }
        }
    
    except Exception as e:
        logger.error(f"Error in process_files: {str(e)}", exc_info=True)
        raise

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve HTML frontend"""
    html_path = Path("index.html")
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Insurance Policy Processing System</h1><p>Upload via POST /process</p>")

@app.post("/process")
async def process_policy(company_name: str = Form(...), policy_file: UploadFile = File(...)):
    """Process policy image"""
    try:
        policy_file_bytes = await policy_file.read()
        if not policy_file_bytes:
            return JSONResponse(status_code=400, content={"error": "Empty file"})
        
        results = process_files(policy_file_bytes, policy_file.filename, policy_file.content_type, company_name)
        return JSONResponse(content=results)
        
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Processing failed: {str(e)}"})

@app.get("/health")
async def health_check():
    """Health check"""
    return JSONResponse(content={"status": "healthy"})

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
