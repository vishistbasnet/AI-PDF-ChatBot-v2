import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model only once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="database")


def search_chunks(question, top_k=5):
    """
    Search the most relevant chunks from ChromaDB.
    """

    collection = client.get_or_create_collection(
        name="pdf_chatbot"
    )

    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # -------------------------------
    # DEBUG OUTPUT
    # -------------------------------
    print("\n========== CHROMADB RESULTS ==========")
    print(results)
    print("======================================\n")

    if not results["documents"] or not results["documents"][0]:
        return []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved_chunks = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        # Skip if metadata is missing
        if metadata is None:
            print("WARNING: Metadata is None")
            continue

        retrieved_chunks.append(
            {
                "text": document,
                "page": metadata.get("page", "N/A"),
                "chunk": metadata.get("chunk", "N/A"),
                "file": metadata.get("file", "Unknown"),
                "score": distance
            }
        )

    retrieved_chunks.sort(key=lambda x: x["score"])

    return retrieved_chunks[:3]