import chromadb
from chromadb.utils import embedding_functions

# client - an object/variable that represents your connection to the service (CHROMADB)
client = chromadb.PersistentClient(path = "./data/chroma_db")

## builds an embedding function powered by (all-MiniLM-L6-v2) model - turns text into numeric vector
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name= "all-MiniLM-L6-v2")

collection = client.get_or_create_collection(name="documents", embedding_function=embedding_fn)

# Insert a batch of text chunks into the collection
def add_chunks(doc_id: str, chunks: list[str]):
    # build a unique ID for each chunk e.g handbook.text_0, handbook.text_1
    ids = [f"{doc_id} {i}"for i in range(len(chunks))]
    metadatas = [{"source": doc_id, "chunk_index": i}for i in range(len(chunks))]
    # inserts evenrything in ChromaDB
    collection.add(documents=chunks, metadatas=metadatas, ids=ids)
    return len(chunks)

def search_chunks(query:str, top_k: int = 4):
    return collection.query(query_texts=[query], n_results=top_k)

def get_documents():
    results = collection.get(include=["metadatas"])
    metadatas = results.get("metadatas", [])
    if metadatas and isinstance(metadatas[0], list):
        flat_metadatas = [item for batch in metadatas for item in batch]
    else:
        flat_metadatas = metadatas
    return sorted({meta["source"] for meta in flat_metadatas if isinstance(meta, dict) and "source" in meta})