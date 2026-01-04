# api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(
    persist_directory="db/chroma_db",
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"}
)
model = ChatOpenAI(model="gpt-4o")

class QueryRequest(BaseModel):
    question: str
    chat_history: list = []

@app.post("/query")
async def query_documents(request: QueryRequest):
    # Convert chat history to LangChain messages
    chat_history = []
    for msg in request.chat_history:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))
    
    # Rewrite question if there's chat history
    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {request.question}")
        ]
        result = model.invoke(messages)
        search_question = result.content.strip()
    else:
        search_question = request.question
    
    # Retrieve relevant documents
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.3}
    )
    docs = retriever.invoke(search_question)
    
    # Create combined input
    combined_input = f"""Based on the following documents, please answer this question: {request.question}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in docs])}

Please provide a clear, helpful answer using only the information from these documents."""
    
    # Get answer
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    
    return {
        "answer": result.content,
        "sources": [doc.metadata.get("source", "Unknown") for doc in docs]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)