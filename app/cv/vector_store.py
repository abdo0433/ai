from embedding import collection, Embedder

embedder = Embedder()

def add_document(doc_id, text, doc_type):
    emb = embedder.embed(text)
    collection.add(
        ids=[doc_id],
        documents=[text],
        metadatas=[{"type": doc_type}],
        embeddings=[emb]
    )

def search_similar(text, top_k=5):
    emb = embedder.embed(text)
    results = collection.query(
        query_embeddings=[emb],
        n_results=top_k
    )
    return results