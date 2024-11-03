import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Import existing RAG system code
from langchain_milvus_rag_chat_api import RAGSystem
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Update initialization
ollama_server = os.getenv("OLLAMA_SERVER", "http://ollama:11434")
milvus_host = os.getenv("MILVUS_HOST", "milvus")
milvus_port = os.getenv("MILVUS_PORT", "19530")

app = FastAPI()

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

@app.post("/process_document", response_model=DocumentOutput)
async def process_document(document: DocumentInput):
    try:
        rag_system.process_document(document.file_path)
        return {"message": f"Document {document.file_path} processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryOutput)
async def query(query_input: QueryInput):
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

@app.on_event("shutdown")
def shutdown_event():
    rag_system.cleanup()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)