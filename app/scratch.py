from app.vectorestore import search_chunks

results = search_chunks("How much is a junior paid?", top_k=3)

for doc, meta, dist in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distance"][0]
):
    
    print(f"[{meta['source']} chunk{meta['chunk_index']}] (distance={dist:.2f})")
    print(doc[:150]," ")