import requests
import duckdb
from sentence_transformers import SentenceTransformer

LAB_URL = "http://<LAB_IP>:8000/ocr"
DB_PATH = "obsidian_brain.db"

def process_and_sync(file_path):
    # 1. Ask Lab for the heavy OCR/Reasoning
    with open(file_path, 'rb') as f:
        response = requests.post(LAB_URL, files={'file': f})
    
    data = response.json() # Contains: 'text', 'topic', 'keywords'

    # 2. Local Indexing into DuckDB
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(data['text']).tolist()
    
    con = duckdb.connect(DB_PATH)
    con.execute("INSERT INTO notes VALUES (?, ?, ?, ?)", 
                [file_path, data['text'], data['topic'], embedding])
    
    print(f"✅ {file_path} processed via Lab and indexed locally.")

if __name__ == "__main__":
    import sys
    process_and_sync(sys.argv[1])
