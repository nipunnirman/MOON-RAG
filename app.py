import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system components"""
    persistent_directory = "db/chroma_db"
    
    if not os.path.exists(persistent_directory):
        st.error("Vector store not found! Please run ingesion.py first.")
        return None, None, None
    
    # Load embeddings and vector store
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    # Create ChatOpenAI model
    model = ChatOpenAI(model="gpt-4o")
    
    # Get document count
    doc_count = db._collection.count()
    
    return db, model, doc_count

def get_standalone_question(user_question, chat_history, model):
    """Reformulate question based on chat history"""
    if not chat_history:
        return user_question
    
    messages = [
        SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
    ] + chat_history + [
        HumanMessage(content=f"New question: {user_question}")
    ]
    
    result = model.invoke(messages)
    return result.content.strip()

def retrieve_documents(db, query, k=3, score_threshold=0.3):
    """Retrieve relevant documents from vector store"""
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": k,
            "score_threshold": score_threshold
        }
    )
    return retriever.invoke(query)

def generate_answer(user_question, relevant_docs, chat_history, model):
    """Generate answer using retrieved documents and chat history"""
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""
    
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    return result.content, relevant_docs

# Main UI
st.title("🤖 RAG Document Q&A System")
st.markdown("Ask questions about your documents and get AI-powered answers!")

# Initialize RAG system
db, model, doc_count = initialize_rag_system()

if db is None:
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.metric("Documents in Database", doc_count)
    
    st.divider()
    
    num_docs = st.slider("Number of documents to retrieve", 1, 10, 3)
    score_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.3, 0.05)
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.markdown("### About")
    st.markdown("This RAG system retrieves relevant documents and generates answers using AI.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 View Source Documents"):
                for i, doc in enumerate(message["sources"], 1):
                    st.markdown(f"**Document {i}:**")
                    st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                    st.markdown(f"*Source: {doc.metadata.get('source', 'Unknown')}*")
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get standalone question
            search_query = get_standalone_question(
                prompt, 
                st.session_state.chat_history, 
                model
            )
            
            # Retrieve documents
            relevant_docs = retrieve_documents(
                db, 
                search_query, 
                k=num_docs, 
                score_threshold=score_threshold
            )
            
            if not relevant_docs:
                response = "I couldn't find any relevant documents to answer your question. Please try rephrasing or ask about something else."
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                # Generate answer
                answer, sources = generate_answer(
                    prompt, 
                    relevant_docs, 
                    st.session_state.chat_history, 
                    model
                )
                
                st.markdown(answer)
                
                # Show sources
                with st.expander("📚 View Source Documents"):
                    for i, doc in enumerate(sources, 1):
                        st.markdown(f"**Document {i}:**")
                        st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                        st.markdown(f"*Source: {doc.metadata.get('source', 'Unknown')}*")
                        st.divider()
                
                # Update chat history
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                st.session_state.chat_history.append(AIMessage(content=answer))
                
                # Add assistant message to display
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 10px;'>
        Powered by LangChain, OpenAI, and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)