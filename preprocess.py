import pandas as pd
import numpy as np
import faiss
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from langchain_huggingface import HuggingFaceEmbeddings

def preprocess_data(csv_file):
    """Loads and preprocesses vehicle data, generates embeddings, and stores them in FAISS."""
    
    df = pd.read_csv(csv_file)

    # Convert Date Columns to Numeric Days Since Today
    today = pd.to_datetime("today")
    for col in ["Last_Service_Date", "Warranty_Expiry_Date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df[col] = (df[col] - today).dt.days

    # Normalize Numerical Columns
    num_cols = ["Mileage", "Vehicle_Age", "Engine_Size", "Odometer_Reading",
                "Insurance_Premium", "Fuel_Efficiency", "Service_History", "Accident_History"]
    
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Encode Categorical Columns
    cat_cols = ["Vehicle_Model", "Maintenance_History", "Fuel_Type",
                "Transmission_Type", "Owner_Type", "Tire_Condition",
                "Brake_Condition", "Battery_Status"]
    
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Save Scaler
    joblib.dump(scaler, "scaler.pkl")

    # Convert Text Data to Embeddings
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    text_data = df["Vehicle_Model"].astype(str) + " " + df["Maintenance_History"].astype(str)
    embeddings = embedder.embed_documents(text_data.tolist())

    # Store embeddings in FAISS
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    faiss.write_index(index, "vehicle_index.faiss")
    df.to_csv("processed_vehicle_data.csv", index=False)

    return df

# Run preprocessing
if __name__ == "__main__":
    preprocess_data("vehicle_data.csv")
