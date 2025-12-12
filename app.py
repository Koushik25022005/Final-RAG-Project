# Main Streamlit interfaces 
import streamlit as st
from utils import process_file, generate_response

def main():
    st.title("ADNOC RAG Assistant")
    st.write("Built by Koushik Sripathi Panditaradhyula")
    uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "jpg", "jpeg", "png"])
    
    if uploaded_file:
        with st.spinner("Processing file..."):
            chunks = process_file(uploaded_file)
            st.success("File processed successfully!")
            st.write(f"Extracted {len(chunks)} text chunks")
        
        query = st.text_input("Ask a question about your document:")
        if query:
            response = generate_response(query, chunks)
            st.write
            st.subheader("Response:")
            st.write(response)

if __name__ == "__main__":
    main()

