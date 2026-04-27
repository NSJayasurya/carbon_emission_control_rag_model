import streamlit as st
from retriever import store_records_in_chromadb, retrieve_similar_records
from generator import generate_response

st.set_page_config(page_title="🚗 AI Vehicle Maintenance Assistant", layout="wide")
st.title("🚗 AI-Powered Vehicle Maintenance Assistant")

# User Input for Query
user_query = st.text_input("🔍 Enter Vehicle Details (e.g., 'Toyota Corolla, 50000 miles, Oil Change Needed'):")

if st.button("🔧 Get Maintenance Suggestion"):
    with st.spinner("⏳ Retrieving similar records..."):
        retrieved_data = retrieve_similar_records(user_query)

    if not retrieved_data.empty:
        st.subheader("📊 Similar Vehicle Records")
        st.write(retrieved_data)

        with st.spinner("🤖 Generating AI-based maintenance suggestion..."):
            response = generate_response(retrieved_data)

        st.success("✅ AI Suggestion Generated!")
        st.subheader("🔧 Maintenance Recommendation")
        st.write(response)
    else:
        st.error("❌ No similar records found. Try a different input.")

# Optional: Allow users to upload their own dataset
uploaded_file = st.file_uploader("📂 Upload a CSV file to store in ChromaDB (Optional)", type=["csv"])

if uploaded_file:
    with st.spinner("⚙️ Storing records in ChromaDB..."):
        result = store_records_in_chromadb(uploaded_file)
    st.success(result)
