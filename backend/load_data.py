import os
import cohere
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import re
from typing import List, Dict
import asyncio

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Initialize clients
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=30
)

co = cohere.Client(COHERE_API_KEY)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk + " " + sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start new chunk with some overlap
            words = current_chunk.split()
            overlap_words = words[-overlap:] if len(words) > overlap else words
            current_chunk = " ".join(overlap_words) + " " + sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def load_book_content() -> List[Dict]:
    """Load and process book content from docs directory"""
    book_chunks = []
    docs_path = "../docs"  # Relative to backend directory
    
    for root, dirs, files in os.walk(docs_path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Extract chapter/section info from path
                    relative_path = os.path.relpath(file_path, docs_path)
                    chapter = relative_path.split('/')[0] if '/' in relative_path else 'introduction'
                    section = os.path.splitext(os.path.basename(file_path))[0]
                    
                    # Clean up content (remove markdown if needed)
                    clean_content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)  # Remove code blocks
                    clean_content = re.sub(r'`.*?`', '', clean_content)  # Remove inline code
                    clean_content = re.sub(r'#.*?\n', '', clean_content)  # Remove headers
                    clean_content = re.sub(r'\*\*.*?\*\*', '', clean_content)  # Remove bold
                    clean_content = re.sub(r'\n+', ' ', clean_content)  # Normalize whitespace
                    
                    # Split into chunks
                    chunks = chunk_text(clean_content, chunk_size=500, overlap=50)
                    
                    for i, chunk in enumerate(chunks):
                        book_chunks.append({
                            'id': f"{chapter}_{section}_{i}",
                            'chapter': chapter,
                            'section': section,
                            'text': chunk
                        })
    
    return book_chunks

def create_qdrant_collection():
    """Create Qdrant collection for book embeddings"""
    try:
        qdrant_client.create_collection(
            collection_name="physical_ai_book",
            vectors_config=models.VectorParams(
                size=1024,  # Cohere embed-english-v3.0 returns 1024-dim vectors
                distance=models.Distance.COSINE
            )
        )
        print("Qdrant collection created successfully")
    except Exception as e:
        print(f"Collection might already exist: {e}")

def embed_and_store_chunks(chunks: List[Dict]):
    """Embed text chunks and store in Qdrant"""
    batch_size = 10  # Process in batches to avoid rate limits
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        
        # Extract texts for embedding
        texts = [chunk['text'] for chunk in batch]
        
        try:
            # Generate embeddings using Cohere
            response = co.embed(
                texts=texts,
                model="embed-english-v3.0",
                input_type="search_document"
            )
            
            embeddings = response.embeddings
            
            # Prepare points for Qdrant
            points = []
            for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                points.append(
                    models.PointStruct(
                        id=i+j,
                        vector=embedding,
                        payload={
                            "id": chunk['id'],
                            "chapter": chunk['chapter'],
                            "section": chunk['section'],
                            "text": chunk['text']
                        }
                    )
                )
            
            # Upload to Qdrant
            qdrant_client.upsert(
                collection_name="physical_ai_book",
                points=points
            )
            
            print(f"Uploaded batch {i//batch_size + 1}: {len(points)} points")
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            continue

def main():
    """Main function to load, embed, and store book content"""
    print("Loading book content...")
    chunks = load_book_content()
    print(f"Loaded {len(chunks)} text chunks")
    
    print("Creating Qdrant collection...")
    create_qdrant_collection()
    
    print("Embedding and storing chunks...")
    embed_and_store_chunks(chunks)
    
    print("Data loading completed!")

if __name__ == "__main__":
    main()