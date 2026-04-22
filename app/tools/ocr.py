# import os
# import json
# import google.generativeai as genai
# from dotenv import load_dotenv

# current_dir = os.path.dirname(os.path.abspath(__file__))
# backend_dir = os.path.dirname(os.path.dirname(current_dir))
# load_dotenv(os.path.join(backend_dir, '.env'))

# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# # Gemini 2.5 Flash natively processes images incredibly fast
# model = genai.GenerativeModel('gemini-2.5-flash')

# def extract_medicine_from_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
#     """
#     Acts as a multi-modal OCR reader. Takes an image of a medicine strip 
#     or prescription and extracts the brand name and dosage.
#     """
    
#     prompt = """
#     You are an expert pharmacist AI. Look at this image of a medicine prescription or list.
#     Identify ALL the medicine brand names and their dosages. 
#     Ignore manufacturing details, addresses, doctor names, and expiry dates.
    
#     If you cannot read anything clearly, set success to false.
    
#     Respond ONLY with a valid JSON object in this exact format:
#     {
#         "success": true or false,
#         "queries": ["Medicine Name 1", "Medicine Name 2", "Medicine Name 3"],
#         "confidence": "high", "medium", or "low",
#         "error_message": "If success is false, explain why (e.g., 'Image too blurry'). Otherwise, null."
#     }
#     """

#     try:
#         # Construct the image payload for Gemini
#         image_part = {
#             "mime_type": mime_type,
#             "data": image_bytes
#         }

#         # Force JSON output for deterministic API behavior
#         response = model.generate_content(
#             [image_part, prompt],
#             generation_config=genai.GenerationConfig(
#                 response_mime_type="application/json",
#                 temperature=0.0 
#             )
#         )
        
#         return json.loads(response.text)

#     except Exception as e:
#         print(f"⚠️ Vision OCR Failed: {e}")
#         return {
#             "success": False, 
#             "query": None, 
#             "confidence": "low", 
#             "error_message": "Vision system offline or image unreadable."
#         }


import os
import base64
import json
from groq import Groq

# Initialize the Groq client
client = Groq() 

def extract_medicine_from_image(image_bytes: bytes, mime_type: str) -> dict:
    """
    Takes raw image bytes, sends them to Groq's Llama 3.2 90B Vision model,
    and returns a structured dictionary containing the extracted medicine names
    based on the strict prompt schema.
    """
    print("🚀 Sending image to Groq Llama 3.2 Vision...")
    
    try:
        # 1. Vision APIs require the image to be converted to a Base64 string
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        image_url = f"data:{mime_type};base64,{base64_image}"

        # 2. Your exact preferred prompt
        prompt = """
        You are an expert pharmacist AI. Look at this image of a medicine prescription or list.
        Identify ALL the medicine brand names and their dosages. 
        Ignore manufacturing details, addresses, doctor names, and expiry dates.
        
        If you cannot read anything clearly, set success to false.
        
        Respond ONLY with a valid JSON object in this exact format:
        {
            "success": true or false,
            "queries": ["Medicine Name 1", "Medicine Name 2", "Medicine Name 3"],
            "confidence": "high", "medium", or "low",
            "error_message": "If success is false, explain why (e.g., 'Image too blurry'). Otherwise, null."
        }
        """

        # 3. Call the Groq API
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct", # Updated Vision Model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            }
                        },
                    ],
                }
            ],
            temperature=0.1, 
            max_tokens=512
        )

        # 4. Extract the raw text from the model's response
        raw_text = response.choices[0].message.content.strip()
        
        # 5. Clean up potential markdown formatting
        if raw_text.startswith("```json"):
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text.replace("```", "").strip()

        # 6. Parse the JSON string directly into a Python dictionary
        extracted_data = json.loads(raw_text)

        # Defensive check: Ensure the model didn't hallucinate a different structure
        if "success" not in extracted_data or "queries" not in extracted_data:
            raise ValueError("Model did not return the requested JSON schema.")

        # Return the exact object the LLM generated!
        return extracted_data

    except json.JSONDecodeError:
        return {
            "success": False,
            "queries": [],
            "confidence": "low",
            "error_message": "Could not parse the handwriting into a clean list. Please take a clearer photo."
        }
    except Exception as e:
        return {
            "success": False,
            "queries": [],
            "confidence": "low",
            "error_message": f"Vision API Error: {str(e)}"
        }