from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

# Initialize database and model
persistent_directory = "db/chroma_db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
model = ChatOpenAI(model="gpt-4o")

# Store conversation history
chat_histories = {}

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('question')
        session_id = data.get('session_id', 'default')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        print(f"\n📝 Received question: {question}")
        
        # Get or create chat history
        if session_id not in chat_histories:
            chat_histories[session_id] = []
        
        chat_history = chat_histories[session_id]
        
        # Reformulate question if there's history
        if chat_history:
            messages = [
                SystemMessage(content="Rewrite the question to be standalone. Return only the rewritten question."),
            ] + chat_history + [
                HumanMessage(content=f"New question: {question}")
            ]
            result = model.invoke(messages)
            search_question = result.content.strip()
            print(f"🔍 Searching for: {search_question}")
        else:
            search_question = question
        
        # Retrieve documents
        retriever = db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(search_question)
        
        print(f"📚 Found {len(docs)} documents")
        
        # Create prompt
        combined_input = f"""Based on the following documents, answer this question: {question}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in docs])}

Provide a clear, helpful answer using only the document information. If you don't have enough information, say so."""
        
        # Get answer
        messages = [
            SystemMessage(content="You are a helpful NASA knowledge assistant that answers questions about the Moon."),
        ] + chat_history + [
            HumanMessage(content=combined_input)
        ]
        
        result = model.invoke(messages)
        answer = result.content
        
        # Update history
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))
        
        # Extract sources
        sources = list(set([doc.metadata.get('source', 'Unknown') for doc in docs]))
        
        print(f"✅ Answer generated successfully")
        
        return jsonify({
            'answer': answer,
            'sources': sources
        })
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    data = request.json
    session_id = data.get('session_id', 'default')
    if session_id in chat_histories:
        chat_histories[session_id] = []
    return jsonify({'message': 'Chat history reset'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Server is running'})

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    print("📡 Server will run on http://127.0.0.1:5000")
    print("🌐 Open index.html in your browser to use the chatbot")
    app.run(debug=True, port=5000)