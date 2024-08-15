#file_path = "https://d18rn0p25nwr6d.cloudfront.net/CIK-0001813756/975b3e9b-268e-4798-a9e4-2a9a7c92dc10.pdf"

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_milvus import Milvus
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility


import hashlib
import uuid

# Server addresses
OLLAMA_SERVER = "http://192.168.1.81:11434"  # Replace with your Ollama server address
MILVUS_HOST = "192.168.1.81"  # Replace with your Milvus server address
MILVUS_PORT = "19530"  # Default Milvus port, change if needed

def main():
    # Connect to Milvus
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    # If the collection already exists with the old schema, you need to drop it and recreate
    # Be cautious with this as it will delete all existing data
    #connections.get_connection().drop_collection("document_store")
    utility.drop_collection("document_store")
    utility.drop_collection("document_tracker")


if __name__ == "__main__":
    main()
