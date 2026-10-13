# rag_demo.py
# ---------------------------------------------------------------
# A very simple RAG (Retrieval-Augmented Generation) demo.
#
# RAG = find the right text first, THEN let the AI answer using it.
#
# The 6 steps:  Load -> Split -> Embed -> Store -> Retrieve -> Answer
# ---------------------------------------------------------------
#
# SETUP (run once in the terminal):
#   pip install langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu openai
#
#   # Only needed for the final answer step (Windows PowerShell):
#   $env:OPENAI_API_KEY = "your-key-here"
#
# RUN (from inside the Introduction_to_RAG folder):
#   python rag_demo.py
# ---------------------------------------------------------------

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
# The question we want to answer
question = "What is a token?"


# Step 1: LOAD the text file
loader = TextLoader("llm_document.txt", encoding="utf-8")
documents = loader.load()

# Step 2: SPLIT the text into small chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Step 3: EMBED (turn text into numbers) using a free local model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 4: STORE the chunks in a FAISS vector store
vector_store = FAISS.from_documents(chunks, embeddings)

# Step 5: RETRIEVE the 3 most relevant chunks for the question
results = vector_store.similarity_search(question, k=3)
context = "\n\n".join(r.page_content for r in results)


# Step 6: ANSWER the question using OpenAI
print("QUESTION:", question)

if os.environ.get("OPENAI_API_KEY"):
    from openai import OpenAI

    client = OpenAI()
    prompt = "Use ONLY this text:\n" + context + "\n\nQuestion: " + question

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    print("\nANSWER:")
    print(response.choices[0].message.content)
else:
    # No API key set, so we just show the text we found.
    print("\n(No OPENAI_API_KEY set. Here is the text RAG found:)\n")
    print(context)
