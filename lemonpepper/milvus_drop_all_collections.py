from pymilvus import connections, utility

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
