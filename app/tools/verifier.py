import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Load Environment Variables
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(current_dir))
load_dotenv(os.path.join(backend_dir, '.env'))

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Use the Flash model for high speed and low latency
model = genai.GenerativeModel('gemini-2.5-flash')

def verify_medicine_safety(user_query: str, identified_salt: str, best_match_data: dict) -> dict:
    """
    Acts as a final Pharmacist Check to ensure the AI didn't hallucinate a dangerous match.
    """
    if not best_match_data:
        return {"is_safe": False, "reason": "No match provided to verify."}

    generic_name = best_match_data.get("generic_name", "Unknown")

    prompt = f"""
    You are a strict, highly accurate Pharmacist AI. Your job is to verify if a generic medicine substitution is medically safe.
    
    User originally asked for: "{user_query}"
    The system identified the core salt as: "{identified_salt}"
    The system is proposing to recommend: "{generic_name}"
    
    Task: 
    1. Check if the proposed generic medicine contains the exact same active ingredients (salts) as the user's request.
    2. Check if the dosages are roughly equivalent or safe. 
    3. If the substitution is UNSAFE, act as a helpful pharmacist and state the exact correct generic chemical composition and dosages the user SHOULD be looking for based on their original query.
    
    Respond ONLY with a valid JSON object in this exact format:
    {{
        "is_safe": true or false,
        "reason": "A one-sentence explanation of why it is safe or unsafe.",
        "suggested_alternative": "If is_safe is false, write the exact generic salt(s) and dosages the user actually needs. If is_safe is true, just output null."
    }}
    """

    try:
        # We enforce JSON output so it is strictly parsable and extremely fast
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.0 # Zero creativity, purely logical
            )
        )
        
        # Parse the JSON string into a Python dictionary
        verification_result = json.loads(response.text)
        return verification_result

    except Exception as e:
        print(f"⚠️ LLM Verification Failed: {e}")
        # If the LLM goes down, default to False to be medically safe
        return {"is_safe": False, "reason": "Verification system offline."}