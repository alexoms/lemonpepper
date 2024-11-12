import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import logging

# Import existing RAG system code
from langchain_milvus_rag_chat_api import RAGSystem
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Update initialization
ollama_server = os.getenv("OLLAMA_SERVER", "http://192.168.1.81:11434")
milvus_host = os.getenv("MILVUS_HOST", "192.168.1.81")
milvus_port = os.getenv("MILVUS_PORT", "19530")



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global rag_system
    logging.info("Starting RAG System...")
    rag_system = RAGSystem(ollama_server, milvus_host, milvus_port)
    yield
    # Shutdown
    logging.info("Shutting down RAG System...")
    rag_system.cleanup()

app = FastAPI(lifespan=lifespan)

rag_system = RAGSystem(ollama_server, milvus_host, milvus_port)

class DocumentInput(BaseModel):
    file_path: str

class QueryInput(BaseModel):
    query: str

class DocumentOutput(BaseModel):
    message: str

class QueryOutput(BaseModel):
    answer: str
    source: str
    source_documents: Optional[List[dict]] = None

class DocumentCleanupInput(BaseModel):
    doc_id: str

@app.post("/process_document", response_model=DocumentOutput)
async def process_document(document: DocumentInput):
    try:
        rag_system.process_document(document.file_path)
        return {"message": f"Document {document.file_path} processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryOutput)
async def query(query_input: QueryInput):
    try:
        result = rag_system.query(query_input.query)
        
        # Convert source_documents to a list of dicts for JSON serialization
        if result['source'] == 'retrieval':
            result['source_documents'] = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in result['source_documents']
            ]
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/clear_all_data")
async def clear_all_data(drop_collections: bool = False):
    """Clear all data from the RAG system"""
    try:
        rag_system.clear_all_data(drop_collections)
        return {"message": "Successfully cleared all data"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear_document")
async def clear_document(cleanup_input: DocumentCleanupInput):
    """Clear specific document from the RAG system"""
    try:
        rag_system.clear_document(cleanup_input.doc_id)
        return {"message": f"Successfully cleared document {cleanup_input.doc_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)