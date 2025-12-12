# Final-RAG-Project

## Objective

To perform the process of Retrieval Augmented Generation( also known as RAG)
for better LLM performance.

## Working 

This is the final RAG LLM project which uses basic python libraries such as torch, langchain, etc.
The RAG model first partitions the uploaded documents into sizeable chunks of chunk size 500 along
with a few overlaps. After it performs metadata extraction with the help of sentence-transformers which
is used for better semantic searchs, text similarity wiht the help of an embedding model 
is used to convert those objects into vectors. 

## Instructions

1. Create a python Virtual Environment usings the following commands below \n
   `python -m venv <venv_name>`\n
   `source <venv_name>/bin/activate`\n
or if you are using Windows\n
   `python -m venv <venv_name>`\n
   `.<venv_name>\Scripts\activate`\n
2. Installing the requirements
    `pip install -r requirements.txt`\n
3. After that, you can run the User Interface with the help of the following command
   `streamlit run app.py`\n
   
