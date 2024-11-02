import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Optional, Dict
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

# Import existing RAG system code
from langchain_milvus_rag_chat_api import RAGSystem

app = FastAPI()

# Security configurations
SECRET_KEY = "your-secret-key"  # Replace with a secure secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# User database (replace with a real database in production)
fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "hashed_password": pwd_context.hash("secret"),
    }
}

# Initialize the RAG system
ollama_server = "http://192.168.1.81:11434"
milvus_host = "192.168.1.81"
milvus_port = "19530"
rag_system = RAGSystem(ollama_server, milvus_host, milvus_port)

# Store user sessions (replace with a database in production)
user_sessions: Dict[str, List[Dict]] = {}

class User(BaseModel):
    username: str

class Token(BaseModel):
    access_token: str
    token_type: str

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

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return User(**user_dict)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = User(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/process_document", response_model=DocumentOutput)
async def process_document(document: DocumentInput, current_user: User = Depends(get_current_user)):
    try:
        rag_system.process_document(document.file_path)
        return {"message": f"Document {document.file_path} processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryOutput)
async def query(query_input: QueryInput, current_user: User = Depends(get_current_user)):
    result = rag_system.query(query_input.query)
    
    # Convert source_documents to a list of dicts for JSON serialization
    if result['source'] == 'retrieval':
        result['source_documents'] = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            } for doc in result['source_documents']
        ]
    
    # Store the query in the user's session
    if current_user.username not in user_sessions:
        user_sessions[current_user.username] = []
    user_sessions[current_user.username].append({
        "query": query_input.query,
        "answer": result['answer'],
        "timestamp": datetime.utcnow().isoformat()
    })
    
    return result

@app.get("/user_queries", response_model=List[Dict])
async def get_user_queries(current_user: User = Depends(get_current_user)):
    return user_sessions.get(current_user.username, [])

@app.on_event("shutdown")
def shutdown_event():
    rag_system.cleanup()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)