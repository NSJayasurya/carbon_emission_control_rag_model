import chromadb
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize ChromaDB Client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Load Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Collection for Vehicle Records
collection = chroma_client.get_or_create_collection("vehicle_maintenance")


def store_records_in_chromadb(csv_file):
    """Load vehicle records from CSV and store them in ChromaDB."""
    df = pd.read_csv(csv_file)
    
    # Convert each row to an embedding
    for index, row in df.iterrows():
        record_text = f"Vehicle: {row['Vehicle_Model']}, Mileage: {row['Mileage']}, Condition: {row['Maintenance_History']}"
        embedding = embedding_model.embed_query(record_text)
        
        collection.add(
            ids=[str(index)],  # Unique ID for each record
            embeddings=[embedding],
            metadatas=[{"Vehicle_Model": row["Vehicle_Model"], "Mileage": row["Mileage"], "Maintenance_History": row["Maintenance_History"]}]
        )

    return "✅ Data Stored in ChromaDB!"


def retrieve_similar_records(query_text, top_k=3):
    """Retrieve the most similar vehicle records from ChromaDB."""
    query_embedding = embedding_model.embed_query(query_text)

    # Search for similar records
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    # Convert to DataFrame
    records = pd.DataFrame(results["metadatas"][0]) if results["metadatas"] else pd.DataFrame()
    return records
