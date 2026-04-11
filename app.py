




import os
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq # Groq import kiya
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *

app = Flask(__name__, template_folder='src/templates', static_folder='src/static')

load_dotenv()

PINECONE_KEY = "pcsk_4gkjJA_Nt4kgSPapMME6aYFwQXfUNBo9U7EjBF9XTUAfFpBYtsboxuJ9ZqaJMhA2nY9XKT"
GROQ_KEY = "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"

os.environ["PINECONE_API_KEY"] = PINECONE_KEY
os.environ["GROQ_API_KEY"] = GROQ_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"

# Pinecone se existing index connect karna
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# --- GROQ MODEL INITIALIZATION ---
chatModel = ChatGroq(
    model="llama-3.3-70b-versatile",
    
    groq_api_key=GROQ_KEY
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"User Input: {msg}")
    
    # RAG Chain call ho rahi hai
    response = rag_chain.invoke({"input": msg})
    
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)