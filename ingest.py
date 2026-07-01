import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

base_path = os.path.dirname(os.path.abspath(__file__))
print("Loading documents...")
loader = DirectoryLoader(base_path, glob="*.txt", loader_cls=TextLoader)
docs = loader.load()

print("Chunking text...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

print("Embedding locally (bypassing Google API)...")
# Using the local sentence-transformers model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

print(f"Success! Ingested {len(chunks)} chunks into ChromaDB.")