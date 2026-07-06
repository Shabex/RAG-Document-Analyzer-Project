import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key = os.getenv("MY_API_KEY"))

## Take a question from the user
def build_prompt(question: str, chunks: list[str], metadatas: list[dict]):
    blocks = []
    for i, (chunk, meta) in enumerate(zip(chunks,metadatas), start=1):
        blocks.append(f"[source {i}: {meta['source']}, chunk {meta['chunk_index']}]\n{chunk}")
    context = "\n\n".join(blocks)

## This bulids and return the complete prompt
    return(
        "You are a helpful assistant. Use the context below to answer the question"
        "in a friendly, clear way. If the context does not contain the answer" 
        "say so politely or say you don't have enough information - do not guess."
        "Reference relevamt sources using [Source N]"
        f"Context:\n{context}\n\n Question: {question}\n\nAnswer:"
    )

def generate_answer(question:str, search_results: dict):
    chunks = search_results["documents"][0]
    metadatas = search_results["metadatas"][0]

    # if the vectores store is empty(nothing uploaded yet), chunks will be an empty list
    # we want
    if not chunks:
        return{"answers": "No documents have been uploaded yet", "source":[]}
    
    prompt = build_prompt(question, chunks, metadatas)
    response = client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        max_tokens = 500,
        messages = [{"role": "user","content": prompt}]
    )

    answer_text = response.choices[0].message.content
    source = [{"source": meta['source'], "chunk_index": meta['chunk_index']} for meta in metadatas]
    return {"answer": answer_text, "source": source}