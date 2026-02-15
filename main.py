import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load env variables
load_dotenv()

# --- Configuration ---
PERSISTENT_DIRECTORY = "db/faiss_index"
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Initialize Database & Model
# Note: Ensure you run 'ingesion.py' first to create the DB!
if os.path.exists(PERSISTENT_DIRECTORY):
    db = FAISS.load_local(PERSISTENT_DIRECTORY, embeddings, allow_dangerous_deserialization=True)
else:
    # Fallback if DB doesn't exist yet, mainly to prevent crash on startup
    print("Warning: FAISS index not found. Please run ingesion.py first.")
    db = None

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# --- Memory Management (Simple In-Memory) ---
# In a production app, use Redis or a SQL DB for history
chat_sessions: dict[str, List[BaseMessage]] = {}

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    session_id: str
    question: str

class Citation(BaseModel):
    title: str
    url: str # We will map file path or source metadata to this

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]

# --- FastAPI App Setup ---
app = FastAPI(title="AstraBot RAG API")

# Allow React Frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"], # Adjust for your React port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized. Run ingestion first.")

    session_id = request.session_id
    user_question = request.question

    # Initialize history for new session
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    history = chat_sessions[session_id]

    # --- Step 1: Contextualize Question ---
    search_question = user_question
    if history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the user's question to be standalone and searchable."),
        ] + history[-4:] + [ # Limit context to last 4 messages to save tokens
            HumanMessage(content=f"New question: {user_question}")
        ]
        reformulated = model.invoke(messages)
        search_question = reformulated.content.strip()
        print(f"Original: {user_question} | Search: {search_question}")

    # --- Step 2: Retrieve Documents ---
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    # Format Citations for Frontend
    citations_list = []
    doc_context = []
    
    for doc in docs:
        doc_context.append(doc.page_content)
        # Extract metadata for citation
        source = doc.metadata.get("source", "Unknown Source")
        citations_list.append(Citation(title=os.path.basename(source), url=source))

    # --- Step 3: Generate Answer ---
    context_text = "\n\n".join(doc_context)
    
    system_prompt = """You are AstraBot, an intelligent astronomy assistant. 
    Answer the user's question based strictly on the provided context documents.
    If the answer is not in the documents, admit you don't know."""

    prompt = f"""Context:
    {context_text}

    Question: 
    {user_question}
    """

    messages = [
        SystemMessage(content=system_prompt)
    ] + history[-4:] + [
        HumanMessage(content=prompt)
    ]

    ai_response = model.invoke(messages)
    answer_text = ai_response.content

    # --- Step 4: Update History ---
    chat_sessions[session_id].append(HumanMessage(content=user_question))
    chat_sessions[session_id].append(AIMessage(content=answer_text))

    return ChatResponse(
        answer=answer_text,
        citations=citations_list
    )

if __name__ == "__main__":
    import uvicorn
    # Run with: python main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)