import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_PATH = "rag/faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# We keep the database loaded in memory here so we don't reload it for every single question
_active_database = None

def load_database():
    """Loads the FAISS database from the hard drive into memory."""
    global _active_database
    if _active_database is None:
        text_to_vector_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        # allow_dangerous_deserialization is required but safe because we created this database ourselves
        _active_database = FAISS.load_local(INDEX_PATH, text_to_vector_model, allow_dangerous_deserialization=True)
    return _active_database

def search_text_database(search_query: str, max_results: int = 3) -> list:
    """Takes a user question, searches the database, and returns the most related text."""
    if not os.path.exists(INDEX_PATH):
        return ["Database missing! Please run 'python3 rag/build_index.py' first."]
        
    database = load_database()
    
    # Ask the database to find the text chunks that best match our search query
    matching_documents = database.similarity_search(search_query, k=max_results)
    
    # We only want the actual English text from the results, not the complex metadata objects
    text_results = [doc.page_content for doc in matching_documents]
    return text_results
