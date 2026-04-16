import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DOCS_DIR = "rag/documents"
INDEX_PATH = "rag/faiss_index"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def build_vector_database():
    print("Step 1: Reading text files...")
    loader = DirectoryLoader(DOCS_DIR, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    print("Step 2: Chopping text into smaller, readable chunks...")
    text_chopper = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_chopper.split_documents(documents)

    print("Step 3: Converting text to numbers (Vectors)...")
    text_to_vector_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    print("Step 4: Creating and saving the FAISS database...")
    vector_database = FAISS.from_documents(text_chunks, text_to_vector_model)
    vector_database.save_local(INDEX_PATH)
    
    print(f"Success! Database saved to {INDEX_PATH}")

if __name__ == "__main__":
    build_vector_database()
