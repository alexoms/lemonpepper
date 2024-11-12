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
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import hashlib
import uuid
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)

def clear_milvus_data(collection_name: str, milvus_host: str, milvus_port: str, drop_collection: bool = False):
    """
    Clear data from Milvus collection.
    
    Args:
        collection_name: Name of the collection to clear
        milvus_host: Milvus server host
        milvus_port: Milvus server port
        drop_collection: If True, drops the entire collection. If False, just deletes all entities
    """
    try:
        # Ensure connection to Milvus
        if not connections.has_connection("default"):
            connections.connect(host=milvus_host, port=milvus_port)
            
        if not utility.has_collection(collection_name):
            print(f"Collection {collection_name} does not exist")
            return
            
        if drop_collection:
            # Drop the entire collection
            utility.drop_collection(collection_name)
            print(f"Collection {collection_name} has been dropped")
        else:
            # Delete all entities but keep the collection
            collection = Collection(collection_name)
            collection.load()
            
            # For document_store collection
            if collection_name == "document_store":
                expr = "text != ''"  # Use the text field instead of pk
            # For document_tracker collection
            elif collection_name == "document_tracker":
                expr = "doc_id != ''"
            else:
                raise ValueError(f"Unknown collection name: {collection_name}")
                
            collection.delete(expr)
            collection.flush()  # Ensure changes are persisted
            print(f"All entities in collection {collection_name} have been deleted")
            
            # Print the number of remaining entities to verify
            print(f"Remaining entities in collection: {collection.num_entities}")
            
    except Exception as e:
        print(f"Error clearing Milvus data: {str(e)}")
        raise
    finally:
        try:
            connections.disconnect("default")
        except Exception as e:
            print(f"Error disconnecting from Milvus: {str(e)}")

def clear_document_by_id(collection_name: str, doc_id: str, milvus_host: str, milvus_port: str):
    """
    Clear data for a specific document from Milvus collection.
    
    Args:
        collection_name: Name of the collection to clear
        doc_id: Document ID to clear
        milvus_host: Milvus server host
        milvus_port: Milvus server port
    """
    try:
        # Ensure connection to Milvus
        if not connections.has_connection("default"):
            connections.connect(host=milvus_host, port=milvus_port)
            
        if not utility.has_collection(collection_name):
            print(f"Collection {collection_name} does not exist")
            return
            
        collection = Collection(collection_name)
        collection.load()
        
        # Delete entities for the specific document
        if collection_name == "document_store":
            # Assuming the doc_id is stored in metadata
            expr = f"doc_id == '{doc_id}'"
        elif collection_name == "document_tracker":
            expr = f"doc_id == '{doc_id}'"
        else:
            raise ValueError(f"Unknown collection name: {collection_name}")
            
        delete_result = collection.delete(expr)
        collection.flush()
        
        print(f"Deleted entities for document {doc_id}")
        print(f"Remaining entities in collection: {collection.num_entities}")
        
    except Exception as e:
        print(f"Error clearing document data: {str(e)}")
        raise
    finally:
        try:
            connections.disconnect("default")
        except Exception as e:
            print(f"Error disconnecting from Milvus: {str(e)}")

def log_collection_status(system: Dict[str, Any]) -> None:
    """Log the current status of collections"""
    try:
        collection = system["collection"]
        doc_tracker = system["document_tracker"]
        
        # Log collection statistics
        logging.info(f"Main collection '{collection.name}' status:")
        logging.info(f"  - Number of entities: {collection.num_entities}")
        
        if utility.has_collection(doc_tracker.name):
            logging.info(f"Document tracker '{doc_tracker.name}' status:")
            logging.info(f"  - Number of entities: {doc_tracker.num_entities}")
            
    except Exception as e:
        logging.error(f"Error logging collection status: {str(e)}")

class RAGSystem:
    def __init__(self, ollama_server: str, milvus_host: str, milvus_port: str):
        logging.info("Initializing RAG System...")
        self.system = initialize_system(ollama_server, milvus_host, milvus_port)
        log_collection_status(self.system)

    def query(self, query: str) -> Dict[str, Any]:
        logging.info(f"Processing query: {query}")
        try:
            ensure_milvus_connection(self.system)
            ensure_collection_loaded(self.system)
            log_collection_status(self.system)
            result = query_system(query, self.system)
            logging.info("Query processed successfully")
            return result
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            raise

    def process_document(self, file_path: str) -> None:
        logging.info(f"Processing document: {file_path}")
        try:
            ensure_milvus_connection(self.system)
            ensure_collection_loaded(self.system)
            process_document(file_path, self.system)
            log_collection_status(self.system)
            logging.info(f"Document processed successfully")
        except Exception as e:
            logging.error(f"Error processing document: {str(e)}")
            raise

    def clear_all_data(self, drop_collections: bool = False):
        """Clear all data from both main collection and document tracker"""
        logging.info("Clearing all data from RAG system...")
        try:
            # Clear main collection
            clear_milvus_data(
                "document_store", 
                self.system["milvus_host"], 
                self.system["milvus_port"],
                drop_collections
            )
            
            # Clear document tracker
            clear_milvus_data(
                "document_tracker", 
                self.system["milvus_host"], 
                self.system["milvus_port"],
                drop_collections
            )
            
            logging.info("Successfully cleared all data")
            
            # If we dropped the collections, we need to reinitialize the system
            if drop_collections:
                logging.info("Reinitializing system after dropping collections...")
                self.system = initialize_system(
                    self.system["ollama_server"],
                    self.system["milvus_host"],
                    self.system["milvus_port"]
                )
                
        except Exception as e:
            logging.error(f"Error clearing data: {str(e)}")
            raise

    def clear_document(self, doc_id: str):
        """Clear specific document from the system"""
        logging.info(f"Clearing document {doc_id} from RAG system...")
        try:
            # Clear from main collection
            clear_document_by_id(
                "document_store",
                doc_id,
                self.system["milvus_host"],
                self.system["milvus_port"]
            )
            
            # Clear from document tracker
            clear_document_by_id(
                "document_tracker",
                doc_id,
                self.system["milvus_host"],
                self.system["milvus_port"]
            )
            
            logging.info(f"Successfully cleared document {doc_id}")
        except Exception as e:
            logging.error(f"Error clearing document: {str(e)}")
            raise

    def cleanup(self) -> None:
        logging.info("Cleaning up RAG System...")
        cleanup_system(self.system)

def ensure_milvus_connection(system: Dict[str, Any]) -> None:
    """Ensure Milvus connection is active, reconnect if necessary"""
    try:
        # Check if connection exists
        if not connections.has_connection("default"):
            connections.connect(
                host=system["milvus_host"],
                port=system["milvus_port"]
            )
    except Exception as e:
        print(f"Error ensuring Milvus connection: {str(e)}")
        # Try to reconnect
        connections.connect(
            host=system["milvus_host"],
            port=system["milvus_port"]
        )

def initialize_system(ollama_server: str, milvus_host: str, milvus_port: str):
    # Connect to Milvus
    connections.connect(host=milvus_host, port=milvus_port)

    # Use Ollama embeddings with the Jina model
    embeddings = OllamaEmbeddings(
        model="jina/jina-embeddings-v2-base-en:latest",
        base_url=ollama_server
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
        connection_args={"host": milvus_host, "port": milvus_port},
    )

    # Create or get the document tracker
    document_tracker = create_document_tracker(embedding_dim)

    # Set up the Ollama LLM
    llm = Ollama(
        model="llama3.1:latest",
        base_url=ollama_server,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        stop=["<|eot_id|>"],
    )

    # Set up the memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Set up the retrieval chain
    retrieval_chain = create_retrieval_chain(llm, vector_store, memory)

    # Set up the general knowledge chain
    general_chain = create_general_knowledge_chain(llm)

    # Add collection to the system dict
    system = {
        "vector_store": vector_store,
        "document_tracker": document_tracker,
        "embedding_dim": embedding_dim,
        "retrieval_chain": retrieval_chain,
        "general_chain": general_chain,
        "memory": memory,
        "collection": collection,
        "ollama_server": ollama_server,
        "milvus_host": milvus_host,
        "milvus_port": milvus_port
    }
    
    ensure_collection_loaded(system)
    return system

def ensure_collection_loaded(system: Dict[str, Any]) -> None:
    """Ensure both main collection and document tracker are loaded in memory"""
    try:
        collection = system["collection"]
        doc_tracker = system["document_tracker"]
        
        # Load collections if they exist
        if utility.has_collection(collection.name):
            collection.load()
            
        if utility.has_collection(doc_tracker.name):
            doc_tracker.load()
            
    except Exception as e:
        print(f"Error ensuring collections are loaded: {str(e)}")
        # Try to load collections anyway
        try:
            collection.load()
            doc_tracker.load()
        except Exception as load_error:
            print(f"Error loading collections: {str(load_error)}")
            raise

def create_retrieval_chain(llm, vector_store, memory):
    retrieval_prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer based on the given context, say "I don't have enough information to answer this question based on the given context."

    {context}

    Chat History:
    {chat_history}

    Human: {question}
    Assistant: """

    RETRIEVAL_PROMPT = PromptTemplate(
        template=retrieval_prompt_template,
        input_variables=["context", "chat_history", "question"]
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": RETRIEVAL_PROMPT},
        return_source_documents=True,
        return_generated_question=True,
    )

def create_general_knowledge_chain(llm):
    general_prompt_template = """You are a helpful AI assistant with broad knowledge on various topics. If the question is about U.S. legislation, legal matters, or Congress, please inform the user that you don't have specific information about that. For all other topics, use your general knowledge to provide the best answer possible.

    Chat History:
    {chat_history}

    Human: {question}
    Assistant: """

    GENERAL_PROMPT = PromptTemplate(
        template=general_prompt_template,
        input_variables=["chat_history", "question"]
    )

    return LLMChain(
        llm=llm,
        prompt=GENERAL_PROMPT,
        output_key="answer"
    )

def process_document(file_path: str, system: Dict[str, Any]) -> None:
    try:    
        vector_store = system["vector_store"]
        doc_tracker = system["document_tracker"]
        embedding_dim = system["embedding_dim"]

        doc_id = hashlib.md5(file_path.encode()).hexdigest()

        if document_exists(doc_tracker, doc_id):
            print(f"Document {file_path} has already been processed. Skipping.")
            return

        loader = PyPDFLoader(file_path)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        
        print("Splitting document...")
        all_splits = text_splitter.split_documents(data)

        texts, metadatas, ids = [], [], []
        
        print("Preparing document chunks...")
        for split in tqdm(all_splits, desc="Processing chunks"):
            split.metadata['doc_id'] = doc_id
            texts.append(split.page_content)
            metadatas.append(split.metadata)
            ids.append(str(uuid.uuid4()))

        print("Adding to vector store...")
        batch_size = 100
        for i in tqdm(range(0, len(texts), batch_size), desc="Inserting batches"):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            vector_store.add_texts(
                texts=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )

        collection = vector_store.col
        collection.flush()
        print(f"Inserted {len(texts)} chunks into the vector store.")
        print(f"Total entities in collection after insertion: {collection.num_entities}")

        mark_document_processed(doc_tracker, doc_id, embedding_dim)
        print(f"Document {file_path} has been processed and added to the vector store.")
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        raise  # Re-raise the exception to be caught by the calling method
    
def query_system(query: str, system: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Ensure collections are loaded before querying
        ensure_collection_loaded(system)
        
        retrieval_chain = system["retrieval_chain"]
        general_chain = system["general_chain"]
        memory = system["memory"]

        retrieval_result = retrieval_chain.invoke({"question": query})
        
        if "I don't have enough information" in retrieval_result['answer']:
            chat_history = memory.load_memory_variables({})["chat_history"]
            general_result = general_chain.predict(question=query, chat_history=chat_history)
            memory.chat_memory.add_user_message(query)
            memory.chat_memory.add_ai_message(general_result)
            return {
                "answer": general_result,
                "source": "general_knowledge"
            }
        else:
            return {
                "answer": retrieval_result['answer'],
                "source": "retrieval",
                "source_documents": retrieval_result['source_documents']
            }
    except Exception as e:
        print(f"Error during query: {str(e)}")
        # Re-ensure collections are loaded and try again
        ensure_collection_loaded(system)
        # If it fails again, let it raise the exception
        return retrieval_chain.invoke({"question": query})

def cleanup_system(system: Dict[str, Any]) -> None:
    system["collection"].release()
    connections.disconnect(alias="default")




def get_embedding_dimension(embeddings):
    sample_embedding = embeddings.embed_query("Sample text")
    return len(sample_embedding)


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

# Modify the main function to use the new class
def main():
    ollama_server = "http://192.168.1.81:11434"
    milvus_host = "192.168.1.81"
    milvus_port = "19530"
    
    rag_system = RAGSystem(ollama_server, milvus_host, milvus_port)
    
    file_path = "https://www.congress.gov/118/bills/hr8785/BILLS-118hr8785ih.pdf"
    print(f"Processing document: {file_path}")
    rag_system.process_document(file_path)

    while True:
        query = input("\nQuery (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        result = rag_system.query(query)
        print("\nAnswer:", result['answer'])
        if result['source'] == 'retrieval':
            print("\nSource Documents:")
            for i, doc in enumerate(result['source_documents'], 1):
                print(f"Document {i}:")
                print(f"Content: {doc.page_content[:100]}...")
                print(f"Metadata: {doc.metadata}")
                print()

    rag_system.cleanup()

if __name__ == "__main__":
    main()