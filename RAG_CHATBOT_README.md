# Physical AI Book RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for the Physical AI & Humanoid Robotics book that answers user questions accurately using the book content.

## Architecture

The system consists of:

1. **Data Layer**: Book content embedded using Cohere embeddings and stored in Qdrant Cloud vector database
2. **Backend**: FastAPI server with Qdrant integration and Groq LLM
3. **Frontend**: Docusaurus React chat component
4. **Agent Logic**: Handles both selected text mode and full book retrieval

## Setup Instructions

### 1. Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with the following:
```env
QDRANT_URL=https://your-qdrant-cluster-url.qdrant.tech:6666
QDRANT_API_KEY=your_qdrant_api_key
COHERE_API_KEY=your_cohere_api_key
GROQ_API_KEY=your_groq_api_key
```

4. Load the book data into Qdrant:
```bash
python load_data.py
```

5. Start the backend server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Frontend Setup

The chat interface is integrated into your Docusaurus site at `/chat` route.

### 3. Usage

1. Make sure the backend server is running on port 8000
2. Access the chat interface at `/chat` on your Docusaurus site
3. Ask questions about Physical AI and Humanoid Robotics
4. You can also select text on any page and ask questions about that specific content

## Features

- **Selected Text Mode**: When you select text on the page, questions will be answered using only that text
- **Full Book Mode**: Questions are answered using the entire book content via vector retrieval
- **Error Handling**: Proper handling of empty results and API failures
- **Responsive UI**: Works well on both desktop and mobile devices

## Technologies Used

- **FastAPI**: Backend server
- **Qdrant**: Vector database for embeddings
- **Cohere**: Text embeddings
- **Groq**: LLM for answer generation
- **React**: Frontend chat interface
- **Docusaurus**: Documentation site integration

## API Endpoints

- `GET /`: Health check
- `POST /chat`: Chat endpoint
  - Input: `{"question": string, "selected_text": string | null}`
  - Output: `{"answer": string}`

## Error Handling

- If Qdrant returns no results: "I don't know. Relevant content not found."
- If API or LLM fails: "Error connecting to server. Please try again."