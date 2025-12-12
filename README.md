# Final-RAG-Project

![RAG-LLM](https://www.google.com/url?sa=i&url=https%3A%2F%2Fapmonitor.com%2Fdde%2Findex.php%2FMain%2FRAGLargeLanguageModel&psig=AOvVaw3z3F0ohHMf4A_8_cu2ahl-&ust=1765612638936000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCPD9jOPJt5EDFQAAAAAdAAAAABAE)

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

1. Create a python Virtual Environment usings the following commands below<br/>
   `python -m venv <venv_name>`<br/>
   `source <venv_name>/bin/activate`<br/>
or if you are using Windows<br/>
   `python -m venv <venv_name>`<br/>
   `.<venv_name>\Scripts\activate`<br/>
2. Installing the requirements<br/>
    `pip install -r requirements.txt`<br/>
3. After that, you can run the User Interface with the help of the following command<br/>
   `streamlit run app.py`<br/>
   
