# rag_demo_langsmith.py
# ---------------------------------------------------------------
# The SAME simple RAG demo, but now LangSmith TRACKS the AI call.
#
# LangSmith records what you send to the model and what comes back,
# so you can SEE it on a website (this is called "tracing").
#
# The only new line is:  client = wrap_openai(OpenAI())
# ---------------------------------------------------------------
#
# SETUP (run once in the terminal):
#   pip install langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu openai langsmith python-dotenv
#
#   # Instead of typing keys in the terminal, just put them in a .env file
#   # (a .env file already sits next to this script - open it and paste your keys).
#
# RUN (from inside the Introduction_to_RAG folder):
#   python rag_demo_langsmith.py
#
# Then open https://smith.langchain.com to see the tracked run.
# ---------------------------------------------------------------

import os
from dotenv import load_dotenv

# Read the keys from the .env file and put them into the environment.
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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


# Step 6: ANSWER the question using OpenAI (and TRACK it with LangSmith)
print("QUESTION:", question)

if os.environ.get("OPENAI_API_KEY"):
    from openai import OpenAI
    from langsmith.wrappers import wrap_openai

    # wrap_openai is the ONLY change needed for LangSmith tracking.
    client = wrap_openai(OpenAI())
    prompt = "Use ONLY this text:\n" + context + "\n\nQuestion: " + question

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    print("\nANSWER:")
    print(response.choices[0].message.content)
    print("\nOpen https://smith.langchain.com to see the tracked run.")
else:
    # No API key set, so we just show the text we found.
    print("\n(No OPENAI_API_KEY set. Here is the text RAG found:)\n")
    print(context)
