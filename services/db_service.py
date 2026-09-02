from langchain_huggingface import HuggingFaceEmbeddings   # مش langchain.embeddings (deprecated)
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    collection_name="data-context",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

def add_docs(chunks):
    """Add documents to the vector store."""
    ids = [chunk.id for chunk in chunks]
    vectorstore.add_documents(chunks, ids=ids)
    return {'added': True}

def update_docs(chunks):
    """Update documents in the vector store."""
    ids = [chunk.id for chunk in chunks]
    vectorstore.update_documents(ids=ids, documents=chunks)
    return {'updated': True}

def delete_docs(ids):
    """Delete documents from the vector store by id."""
    vectorstore.delete(ids=ids)
    return {'deleted': True}