#file_path = "https://d18rn0p25nwr6d.cloudfront.net/CIK-0001813756/975b3e9b-268e-4798-a9e4-2a9a7c92dc10.pdf"
#file_path = "https://www.congress.gov/117/plaws/publ328/PLAW-117publ328.pdf"
#file_path = "https://www.congress.gov/115/plaws/publ141/PLAW-115publ141.pdf"
#file_path = "https://www.congress.gov/116/plaws/publ260/PLAW-116publ260.pdf"
#file_path = "https://www.congress.gov/118/bills/hr2882/BILLS-118hr2882enr.pdf"
#file_path="https://www.congress.gov/118/bills/hr8785/BILLS-118hr8785ih.pdf"
#file_path = "https://www.congress.gov/118/bills/hr7024/BILLS-118hr7024pcs.pdf"

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
from tqdm import tqdm

import hashlib
import uuid

# Server addresses
OLLAMA_SERVER = "http://192.168.1.81:11434"  # Replace with your Ollama server address
MILVUS_HOST = "192.168.1.81"  # Replace with your Milvus server address
MILVUS_PORT = "19530"  # Default Milvus port, change if needed


def get_embedding_dimension(embeddings):
    sample_embedding = embeddings.embed_query("Sample text")
    return len(sample_embedding)

def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)  # Drop existing collection to recreate with correct schema

    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100), 
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535), 
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)]
    schema = CollectionSchema(fields, f"{collection_name} for document storage")
    collection = Collection(collection_name, schema)
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 1024}
    }
    collection.create_index("vector", index_params)
    collection.flush()
    return collection

# Modify the create_document_tracker function
def create_document_tracker(embedding_dim):
    collection_name = "document_tracker"
    if utility.has_collection(collection_name):
        return Collection(collection_name)

    fields = [
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
        FieldSchema(name="processed", dtype=DataType.BOOL),
        FieldSchema(name="dummy_vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
    ]
    schema = CollectionSchema(fields, "Document tracker for deduplication")
    document_tracker = Collection(collection_name, schema)
    document_tracker.create_index(field_name="dummy_vector", index_params={"index_type": "FLAT", "metric_type": "L2", "params": {}})
    return document_tracker

def document_exists(doc_tracker, doc_id):
    doc_tracker.load()
    results = doc_tracker.query(expr=f'doc_id == "{doc_id}"', output_fields=["processed"])
    return len(results) > 0 and results[0]['processed']

# Modify the mark_document_processed function
def mark_document_processed(doc_tracker, doc_id, embedding_dim):
    doc_tracker.insert([
        [doc_id],  # doc_id
        [True],    # processed
        [[0.0] * embedding_dim]  # dummy_vector with correct dimensions
    ])
    doc_tracker.flush()

def process_document(file_path, vector_store, doc_tracker, embedding_dim):
    # Generate a unique identifier for the document
    doc_id = hashlib.md5(file_path.encode()).hexdigest()

    if document_exists(doc_tracker, doc_id):
        print(f"Document {file_path} has already been processed. Skipping.")
        return

    # Load and process the document
    loader = PyPDFLoader(file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    
    print("Splitting document...")
    all_splits = text_splitter.split_documents(data)

    # Add metadata to each split and generate IDs
    texts = []
    metadatas = []
    ids = []
    
    print("Preparing document chunks...")
    for split in tqdm(all_splits, desc="Processing chunks"):
        split.metadata['doc_id'] = doc_id
        texts.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(str(uuid.uuid4()))  # Generate a unique ID for each split

    # Add to vector store
    print("Adding to vector store...")
    batch_size = 100  # Adjust this based on your system's capabilities
    for i in tqdm(range(0, len(texts), batch_size), desc="Inserting batches"):
        batch_texts = texts[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        vector_store.add_texts(
            texts=batch_texts,
            metadatas=batch_metadatas,
            ids=batch_ids
        )

    # Verify insertion
    collection = vector_store.col
    collection.flush()  # Ensure all data is flushed to disk
    print(f"Inserted {len(texts)} chunks into the vector store.")
    print(f"Total entities in collection after insertion: {collection.num_entities}")

    # Mark document as processed
    mark_document_processed(doc_tracker, doc_id, embedding_dim)
    print(f"Document {file_path} has been processed and added to the vector store.")


def get_or_create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        print(f"Loaded existing collection '{collection_name}' with {collection.num_entities} entities.")
    else:
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
        schema = CollectionSchema(fields, f"{collection_name} for document storage")
        collection = Collection(collection_name, schema)
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 1024}
        }
        collection.create_index("vector", index_params)
        print(f"Created new collection '{collection_name}'.")
    
    collection.load()
    return collection

def main():
    # Connect to Milvus
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

    # Use Ollama embeddings with the Jina model
    embeddings = OllamaEmbeddings(
        model="jina/jina-embeddings-v2-base-en:latest",
        base_url=OLLAMA_SERVER
    )

    # Get the embedding dimension
    embedding_dim = get_embedding_dimension(embeddings)

    # Get or create the vector store collection
    collection_name = "document_store"
    collection = get_or_create_milvus_collection(collection_name, embedding_dim)

    # Create the vector store
    vector_store = Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
    )

    # Create or get the document tracker
    document_tracker = create_document_tracker(embedding_dim)

    # Process the document (if needed)
    #file_path = "https://www.congress.gov/118/bills/hr8785/BILLS-118hr8785ih.pdf"
    file_path = "https://www.congress.gov/118/bills/hr7024/BILLS-118hr7024pcs.pdf"
    print(f"Processing document: {file_path}")
    process_document(file_path, vector_store, document_tracker, embedding_dim)

    # Verify the number of entities in the collection after processing
    print(f"Total entities in collection after processing: {collection.num_entities}")

    # Set up the Ollama LLM
    llm = Ollama(
        model="llama3.1:latest",
        base_url=OLLAMA_SERVER,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        stop=["<|eot_id|>"],
    )

    # Define a custom prompt template
    custom_prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer: """

    PROMPT = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )

    # Set up the RetrievalQA chain with the custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    # Get user query and run the chain
    while True:
        query = input("\nQuery (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        result = qa_chain.invoke({"query": query})
        print("\nAnswer:", result['result'])
        print("\nSource Documents:")
        for i, doc in enumerate(result['source_documents'], 1):
            print(f"Document {i}:")
            print(f"Content: {doc.page_content[:100]}...")  # Print first 100 characters
            print(f"Metadata: {doc.metadata}")
            print()

    # Clean up
    collection.release()
    connections.disconnect(alias="default")

if __name__ == "__main__":
    main()