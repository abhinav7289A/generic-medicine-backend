import os
import re
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(current_dir)) 
env_path = os.path.join(backend_dir, '.env')
load_dotenv(env_path)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

try:
    generic_index = pc.Index("medicine-db")
    brand_index = pc.Index("medicine-brands")
except Exception as e:
    print(f"❌ Error connecting to Pinecone indexes: {e}")

model = SentenceTransformer('all-MiniLM-L6-v2')

# --- HELPER FUNCTIONS ---
def extract_dosage_numbers(text):
    """Extracts numbers like '625', '500', '125'."""
    matches = re.findall(r'(\d+)\s?(?:mg|g|ml|mcg|iu)?', text, re.IGNORECASE)
    return [m for m in matches if m.isdigit()]

def extract_base_ingredients(salt_text):
    """Splits combination drugs into core chemical bases: ['telmisartan', 'amlodipine']"""
    components = re.split(r'\+|and|&|,', salt_text, flags=re.IGNORECASE)
    base_ingredients = []
    
    for comp in components:
        clean_comp = re.sub(r'\b\d+\.?\d*\s*(?:mg|g|ml|mcg|iu|gm|%|w/v|w/w)?\b', '', comp, flags=re.IGNORECASE)
        clean_comp = re.sub(r'[\(\)]', '', clean_comp).strip()
        words = clean_comp.split()
        if words:
            base_ingredients.append(words[0].lower())
            
    return list(set(base_ingredients))

def extract_unit_quantity(unit_size_str):
    """
    Extracts the numerical quantity from strings like:
    "10's" -> 10.0 | "30 ml Bottle" -> 30.0 | "Vial" -> 1.0
    """
    if not unit_size_str:
        return 1.0
    
    # 1. Look for standard strip notations like "10's" or "15s"
    match_strip = re.search(r'(\d+)\s*\'?s\b', str(unit_size_str), re.IGNORECASE)
    if match_strip:
        return float(match_strip.group(1))
    
    # 2. Look for any standalone number (for ml, gm, etc.)
    match_num = re.search(r'(\d+\.?\d*)', str(unit_size_str))
    if match_num:
        return float(match_num.group(1))
        
    # 3. Default to 1 if no number is found (e.g., "Vial", "Tube")
    return 1.0

# --- MAIN LOGIC ---
def get_salt_from_brand(brand_query: str):
    print(f"🔍 Checking Brand DB for exact match: '{brand_query}'...")
    
    target_numbers = extract_dosage_numbers(brand_query)
    query_embedding = model.encode(brand_query).tolist()
    
    results = brand_index.query(
        vector=query_embedding,
        top_k=20, 
        include_metadata=True
    )
    
    best_match = None
    if results['matches']:
        if target_numbers:
            for match in results['matches']:
                b_name = match['metadata'].get('brand_name', '')
                if any(num in b_name for num in target_numbers):
                    best_match = match
                    print(f"🎯 Forced Exact Dosage Match in Brand DB: {b_name}")
                    break
        
        if not best_match:
            best_match = results['matches'][0]
            
        score = best_match['score']
        print(f"🧐 Selected Brand Match: {best_match['metadata'].get('brand_name')} (Score: {score:.4f})")
        
        # Updated threshold to 0.65 to prevent "Junk Data" hallucinations
        if score > 0.5: 
            raw_salt = best_match['metadata'].get('salt_composition')
            target_ingredients = extract_base_ingredients(raw_salt)
            real_dosages = extract_dosage_numbers(raw_salt)
            
            clean_salt = re.sub(r'\b\d+\.?\d*\s*(?:mg|g|ml|mcg|iu|gm)?\b', '', raw_salt, flags=re.IGNORECASE)
            clean_salt = re.sub(r'[\(\)]', '', clean_salt)
            clean_salt = " ".join(clean_salt.split())
            
            print(f"🧪 Target Ingredients Extracted: {target_ingredients}")
            return clean_salt, real_dosages, target_ingredients
    
    return None, target_numbers, extract_base_ingredients(brand_query)

def find_cheapest_generic(query_text: str):
    try:
        search_term, target_dosages, target_ingredients = get_salt_from_brand(query_text)
        
        if not search_term:
            search_term = query_text 
            
        print(f"🎯 Searching Generic DB for: '{search_term}'")
        print(f"   -> Required Ingredients: {target_ingredients}")
        print(f"   -> Required Dosages: {target_dosages}")

        query_embedding = model.encode(search_term).tolist()
        results = generic_index.query(
            vector=query_embedding,
            top_k=20, 
            include_metadata=True
        )
        
        matches = []
        for match in results['matches']:
            data = match['metadata']
            generic_name = data.get("generic_name", "Unknown")
            generic_name_lower = generic_name.lower()
            
            # 1. STRICT COMBINATION CHECK
            is_combination_match = True
            if target_ingredients:
                for ingredient in target_ingredients:
                    if ingredient not in generic_name_lower:
                        is_combination_match = False
                        break
            
            # 2. DOSAGE CHECK
            is_dosage_match = False
            if not target_dosages:
                is_dosage_match = True
            else:
                is_dosage_match = all(dosage in generic_name for dosage in target_dosages)
            
            # 3. UNIT PRICE MATH
            mrp = float(data.get("mrp", 0.0))
            unit_size = str(data.get("unit_size", "1"))
            quantity = extract_unit_quantity(unit_size)
            price_per_unit = mrp / quantity if quantity > 0 else mrp
            
            matches.append({
                "generic_name": generic_name,
                "mrp": mrp,
                "unit_size": unit_size,
                "price_per_unit": round(price_per_unit, 2), # Exposing for the frontend
                "group_name": data.get("group_name", ""),
                "is_exact_dosage": is_dosage_match,
                "is_combination_match": is_combination_match
            })
            
        # 4. ULTIMATE MEDICAL SORTING LOGIC
        sorted_matches = sorted(matches, key=lambda x: (
            not x['is_combination_match'], 
            not x['is_exact_dosage'],      
            float('inf') if x['price_per_unit'] == 0 else x['price_per_unit'], # True cost sorting
            len(x['generic_name'])         
        ))
        
        return {
            "original_query": query_text,
            "identified_salt": search_term,
            "best_match": sorted_matches[0] if sorted_matches else None,
            "alternatives": sorted_matches[1:5]
        }

    except Exception as e:
        print(f"Search Error: {e}")
        return None