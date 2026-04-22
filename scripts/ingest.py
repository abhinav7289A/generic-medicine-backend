import os
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from dotenv import load_dotenv

# --- 1. ROBUST PATH CONFIGURATION ---
# Get the absolute path of THIS script (backend/scripts/ingest.py)
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
backend_dir = os.path.dirname(scripts_dir)  # Go up one level to 'backend'

# Construct the exact paths
env_path = os.path.join(backend_dir, '.env')
csv_path = os.path.join(backend_dir, 'data', 'jan_aushadhi.csv')

print(f"🔍 Looking for .env file at: {env_path}")
print(f"🔍 Looking for CSV file at: {csv_path}")

# --- 2. LOAD SECRETS ---
# Load the .env file
loaded = load_dotenv(env_path)

if not loaded:
    print("⚠️ WARNING: .env file was not found or is empty.")
    print("👉 Tip: Check if your file is named '.env.txt' by mistake (Windows hides extensions).")

# Get API Key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError(
        "❌ CRITICAL ERROR: 'PINECONE_API_KEY' not found.\n"
        "1. Check if .env file exists at the path printed above.\n"
        "2. Open .env and ensure it says: PINECONE_API_KEY=your_key_here"
    )

# --- 3. INGESTION LOGIC ---
INDEX_NAME = "medicine-db"

def ingest_data():
    print("\n🚀 Starting Ingestion Process...")

    # Initialize Pinecone
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return

    # Create Index if needed
    existing_indexes = [i.name for i in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        print(f"⚙️ Index '{INDEX_NAME}' not found. Creating it...")
        try:
            pc.create_index(
                name=INDEX_NAME,
                dimension=384, # Matches 'all-MiniLM-L6-v2'
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Wait a moment for index to initialize
            import time
            time.sleep(10)
        except Exception as e:
            print(f"❌ Error creating index: {e}")
            return
    else:
        print(f"✅ Index '{INDEX_NAME}' already exists.")
    
    index = pc.Index(INDEX_NAME)

    # Load AI Model
    print("🧠 Loading AI Model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Read CSV
    print("📂 Reading CSV data...")
    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV file not found at {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['Generic Name'])
        df = df.fillna('')
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    print(f"📊 Found {len(df)} medicines to process.")

    # Process and Upload
    batch_size = 100
    vectors_to_upload = []
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Vectorizing"):
        
        # Create context text
        text_to_embed = f"{row['Generic Name']} {row['Group Name']}"
        
        # Create Vector
        embedding = model.encode(text_to_embed).tolist()
        
        # Create Metadata
        metadata = {
            "drug_code": str(row['Drug Code']),
            "generic_name": str(row['Generic Name']),
            "unit_size": str(row['Unit Size']),
            "mrp": float(row['MRP']) if row['MRP'] else 0.0,
            "group_name": str(row['Group Name'])
        }
        
        vectors_to_upload.append({
            "id": str(row['Drug Code']),
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

    print("\n✅ Ingestion Complete! Your database is ready.")

if __name__ == "__main__":
    ingest_data()