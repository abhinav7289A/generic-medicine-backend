import os
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from dotenv import load_dotenv

# --- 1. ROBUST PATH CONFIGURATION ---
# Get absolute paths to ensure it works from any terminal location
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
backend_dir = os.path.dirname(scripts_dir)
env_path = os.path.join(backend_dir, '.env')
csv_path = os.path.join(backend_dir, 'data', 'brands.csv') # Ensure file is named brands.csv

# --- 2. LOAD SECRETS ---
load_dotenv(env_path)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("❌ CRITICAL ERROR: PINECONE_API_KEY not found. Check your .env file.")

# --- 3. CONFIGURATION ---
INDEX_NAME = "medicine-brands"

def ingest_brands():
    print("\n🚀 Starting Brand Ingestion...")

    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create Index if needed
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"⚙️ Creating Index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    index = pc.Index(INDEX_NAME)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # --- 4. READ & CLEAN DATA ---
    print(f"📂 Reading CSV from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Error: Could not find 'brands.csv' in backend/data/")
        return

    # --- 5. EXACT COLUMN MAPPING ---
    # We use the exact names you provided
    col_name = 'name'
    col_manufacturer = 'manufacturer_name'
    col_price = 'price(â‚¹)'  # Handling the special character
    col_pack_size = 'pack_size_label'
    
    # Identify composition columns dynamically (short_composition1, short_composition2)
    comp_cols = ['short_composition1', 'short_composition2']

    print(f"📊 Processing {len(df)} brands...")

    batch_size = 100
    vectors_to_upload = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Vectorizing"):
        
        # A. MERGE COMPOSITIONS
        # Combine short_composition1 + short_composition2
        salts = []
        for col in comp_cols:
            if col in df.columns:
                val = str(row[col])
                if val and val.lower() != "nan" and val.strip() != "":
                    salts.append(val.strip())
        
        full_salt_composition = " + ".join(salts)
        
        # If no salt info exists, skip this row (useless for our purpose)
        if not full_salt_composition:
            continue

        # B. PREPARE DATA
        brand_name = str(row[col_name]).strip()
        manufacturer = str(row[col_manufacturer]).strip()
        pack_size = str(row[col_pack_size]).strip()
        
        # Handle Price safely (remove symbols if any remain, though pandas usually handles it)
        try:
            price_raw = row[col_price]
            # Clean string if it contains non-numeric characters (optional safety)
            # price = float(str(price_raw).replace('â‚¹', '').strip()) 
            price = float(price_raw)
        except:
            price = 0.0

        # C. EMBEDDING
        # We embed: "Brand Name + Manufacturer" so the search is specific
        text_to_embed = f"{brand_name} {manufacturer}"
        embedding = model.encode(text_to_embed).tolist()

        # D. METADATA
        # This is what we retrieve to verify the salt
        metadata = {
            "brand_name": brand_name,
            "salt_composition": full_salt_composition, # The critical "Truth"
            "manufacturer": manufacturer,
            "pack_size": pack_size,
            "price": price
        }

        vectors_to_upload.append({
            "id": f"brand_{i}", 
            "values": embedding,
            "metadata": metadata
        })

        # Upload Batch
        if len(vectors_to_upload) >= batch_size:
            try:
                index.upsert(vectors=vectors_to_upload)
                vectors_to_upload = []
            except Exception as e:
                print(f"⚠️ Error uploading batch: {e}")

    # Upload remaining
    if vectors_to_upload:
        index.upsert(vectors=vectors_to_upload)

    print("\n✅ Brand Database Ingested Successfully!")

if __name__ == "__main__":
    ingest_brands()