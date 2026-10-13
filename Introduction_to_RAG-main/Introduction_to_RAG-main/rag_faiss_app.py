import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

PDF_PATH = "spotify_web_app_architecture.pdf"
FAISS_DB_PATH = "faiss_index"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def create_vector_db():
    print("Loading PDF...")

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print("Creating FAISS vector DB...")

    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(FAISS_DB_PATH)

    print("FAISS vector DB saved locally.")
    print(f"Total chunks stored: {len(chunks)}")


def load_vector_db():
    if not os.path.exists(FAISS_DB_PATH):
        create_vector_db()

    return FAISS.load_local(
        FAISS_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def ask_question(question, vector_db):
    docs = vector_db.similarity_search(question, k=3)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template("""
You are a helpful RAG assistant.

Answer the user's question only using the given context.
If the answer is not available in the context, say:
"I don't know from the provided document."

Context:
{context}

Question:
{question}

Answer:
""")

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0
    )

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content


def main():
    vector_db = load_vector_db()

    print("\nRAG app is ready.")
    print("Ask questions from your PDF.")
    print("Type 'exit' to stop.")

    while True:
        question = input("\nAsk question: ")

        if question.lower() in ["exit", "quit"]:
            print("Bye!")
            break

        answer = ask_question(question, vector_db)

        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()