from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from app.tools.database import find_cheapest_generic
from app.tools.verifier import verify_medicine_safety
from app.tools.ocr import extract_medicine_from_image # <-- Import the Vision Tool

app = FastAPI(title="Generic Medicine Finder API")

class MedicineQuery(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "Agentic AI Backend is Running!"}

@app.post("/search")
def search_medicine(payload: MedicineQuery):
    print(f"📝 Received raw payload: {payload}")
    
    # 1. THE BUG FIX: Extract the actual string from the Pydantic object
    raw_text = payload.query 
    
    # 2. MULTIPLE MEDICINES: Split the string by commas to allow batch searching
    # Example: "Dolo 650, Telma 40" -> ["Dolo 650", "Telma 40"]
    queries = [q.strip() for q in raw_text.split(",") if q.strip()]
    
    final_text_results = []
    
    # 3. Loop through every medicine typed in the box
    for medicine_name in queries:
        print(f"🔍 Processing text query: '{medicine_name}'...")
        
        # Run Pinecone Search (Passing the STRING, not the object!)
        db_results = find_cheapest_generic(medicine_name)
        
        if not db_results or not db_results.get("best_match"):
            final_text_results.append({
                "original_query": medicine_name,
                "error": "No generic alternative found in database."
            })
            continue
            
        # Run Gemini Safety Verification
        verification = verify_medicine_safety(
            user_query=db_results["original_query"],
            identified_salt=db_results["identified_salt"],
            best_match_data=db_results["best_match"]
        )
        
        db_results["verification"] = verification
        
        if not verification.get("is_safe"):
            db_results["warning"] = "The AI Pharmacist flagged this substitution. Please consult a doctor."
            
        final_text_results.append(db_results)
        
    # 4. Return the exact same JSON structure as the Vision endpoint
    return {
        "search_count": len(queries),
        "results": final_text_results
    }

@app.post("/vision-search")
async def search_medicine_from_image(file: UploadFile = File(...)):
    """
    Receives an image, extracts MULTIPLE medicines using VLM OCR, 
    and automatically runs each through the search and verification pipeline.
    """
    print(f"📸 Received image: {file.filename} ({file.content_type})")
    
    # 1. Read the image bytes
    image_bytes = await file.read()
    
    # 2. Run the Vision OCR extraction
    ocr_result = extract_medicine_from_image(image_bytes, file.content_type)
    print(f"👁️ OCR Result: {ocr_result}")
    
    if not ocr_result.get("success") or not ocr_result.get("queries"):
        raise HTTPException(
            status_code=400, 
            detail=ocr_result.get("error_message", "Could not read medicines from image.")
        )
        
    extracted_queries = ocr_result["queries"]
    final_prescription_results = []
    
    # 3. Loop through every medicine found on the prescription
    for query in extracted_queries:
        print(f"\\n🔄 Processing extracted query: '{query}'...")
        
        # Run Pinecone Search
        db_results = find_cheapest_generic(query)
        
        if not db_results or not db_results.get("best_match"):
            # If we can't find one specific medicine, we append an error for just that item
            # so the rest of the list doesn't crash.
            final_prescription_results.append({
                "original_query": query,
                "error": "No generic alternative found in database."
            })
            continue
            
        # Run Gemini Safety Verification
        verification = verify_medicine_safety(
            user_query=db_results["original_query"],
            identified_salt=db_results["identified_salt"],
            best_match_data=db_results["best_match"]
        )
        
        db_results["verification"] = verification
        db_results["extracted_via_ocr"] = True 
        
        if not verification.get("is_safe"):
            db_results["warning"] = "The AI Pharmacist flagged this substitution. Please consult a doctor."
            
        # Add the completed, verified medicine to our final list
        final_prescription_results.append(db_results)
        
    return {
        "prescription_count": len(extracted_queries),
        "results": final_prescription_results
    }