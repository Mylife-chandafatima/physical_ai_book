from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.http import models
import cohere
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Physical AI Book RAG Chatbot")

# Add CORS middleware for Docusaurus frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Docusaurus domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration - these come from environment variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize clients
qdrant_client = None
co = None
groq_client = None

# Check if all required API keys are provided
if QDRANT_URL and QDRANT_API_KEY:
    try:
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=10
        )
        print("Qdrant client initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize Qdrant client: {e}")

if COHERE_API_KEY:
    try:
        co = cohere.Client(COHERE_API_KEY)
        print("Cohere client initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize Cohere client: {e}")

if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("Groq client initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize Groq client: {e}")

# Check if all clients are available
all_clients_available = all([qdrant_client, co, groq_client])

if all_clients_available:
    print("All clients initialized successfully. Full RAG functionality available.")
else:
    print("Not all clients are available. Some functionality may be limited.")
    print(f"Qdrant client available: {qdrant_client is not None}")
    print(f"Cohere client available: {co is not None}")
    print(f"Groq client available: {groq_client is not None}")


class ChatRequest(BaseModel):
    question: str
    selected_text: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str

@app.get("/")
def read_root():
    return {"message": "Physical AI Book RAG Chatbot API"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # If selected_text is provided, use only that text
        if request.selected_text:
            answer = await process_selected_text_mode(request.question, request.selected_text)
        else:
            # Otherwise use full book mode with retrieval
            answer = await process_full_book_mode(request.question)
        
        return ChatResponse(answer=answer)
    
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Error connecting to server. Please try again.")

async def process_selected_text_mode(question: str, selected_text: str) -> str:
    """Process question using only the selected text"""
    try:
        if groq_client:
            # Use Groq to generate answer based on selected text
            prompt = f"""
            You are an expert on Physical AI and Humanoid Robotics. Answer the question based ONLY on the provided text.
            
            Question: {question}
            
            Provided Text: {selected_text}
            
            Answer the question using only the information from the provided text. 
            If the answer cannot be found in the provided text, respond with: "I don't know. Relevant content not found."
            """
            
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="mixtral-8x7b-32768",  # Using Mixtral model for good performance
                temperature=0,
                max_tokens=1000
            )
            
            return chat_completion.choices[0].message.content
        else:
            # Fallback if Groq is not available
            return f"Selected text mode: '{selected_text[:100]}...' - This feature requires a valid GROQ_API_KEY."
    
    except Exception as e:
        print(f"Error in selected text mode: {str(e)}")
        return "Error connecting to server. Please try again."

async def process_full_book_mode(question: str) -> str:
    """Process question using full book retrieval"""
    try:
        if not all_clients_available:
            return "This feature requires valid API keys for Qdrant, Cohere, and Groq. Please check your configuration."
        
        # Embed the question using Cohere
        response = co.embed(
            texts=[question],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        question_embedding = response.embeddings[0]
        
        # Query Qdrant for top 5 most relevant chunks
        search_results = qdrant_client.search(
            collection_name="physical_ai_book",
            query_vector=question_embedding,
            limit=5,
            with_payload=True
        )
        
        if not search_results:
            return "I don't know. Relevant content not found."
        
        # Extract text from results to form context
        context_texts = []
        for result in search_results:
            if result.payload and 'text' in result.payload:
                context_texts.append(result.payload['text'])
        
        if not context_texts:
            return "I don't know. Relevant content not found."
        
        # Combine context
        context = "\n\n".join(context_texts)
        
        # Use Groq to generate answer based on retrieved context
        prompt = f"""
        You are an expert on Physical AI and Humanoid Robotics. Answer the question based ONLY on the provided context from the book.
        
        Question: {question}
        
        Context from the book:
        {context}
        
        Answer the question using only the information from the provided context. 
        If the answer cannot be found in the context, respond with: "I don't know. Relevant content not found."
        Be concise but comprehensive in your answer.
        """
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0,
            max_tokens=1000
        )
        
        return chat_completion.choices[0].message.content
    
    except Exception as e:
        print(f"Error in full book mode: {str(e)}")
        return "Error connecting to server. Please try again."

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)