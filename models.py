import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

# Retrieve API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Check if API key is loaded (optional debugging)
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please check your .env file.")

# Initialize AI models
def create_chat_groq_model(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2):
    return ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        cache=False,
        api_key=GROQ_API_KEY  # ✅ API key passed here
    )

def create_hugging_face_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)

# Load models
chat_model = create_chat_groq_model()
embedding_model = create_hugging_face_embedding_model()

def process_data(df: pd.DataFrame):
    """
    Process vehicle maintenance data using the AI model.
    """
    results = []
    for _, row in df.iterrows():
        input_text = f"Vehicle: {row['Vehicle_Model']}, Mileage: {row['Mileage']}, Condition: {row['Maintenance_History']}"
        
        try:
            response = chat_model.invoke(input_text)  # 🔥 Calls the Groq API
            results.append({"Vehicle": row["Vehicle_Model"], "Prediction": response})
        except Exception as e:
            print(f"Error processing row {row['Vehicle_Model']}: {e}")
            results.append({"Vehicle": row["Vehicle_Model"], "Prediction": "Error"})

    return pd.DataFrame(results)
