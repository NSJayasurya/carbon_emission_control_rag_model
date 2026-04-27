from langchain_groq import ChatGroq

def create_chat_model(model="mixtral-8x7b-32768", temperature=0):
    """Load the LLM model."""
    return ChatGroq(model=model, temperature=temperature)

def generate_response(retrieved_data):
    """Generate text response using LLM based on retrieved records."""
    model = create_chat_model()

    prompt = f"""You are an expert vehicle maintenance assistant.
    Based on the following vehicle records, suggest maintenance actions:

    {retrieved_data.to_string(index=False)}

    Provide a concise and actionable response."""

    response = model.invoke(prompt)
    return response
