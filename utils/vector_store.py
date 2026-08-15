import uuid
import chromadb

# Connect to ChromaDB
client = chromadb.PersistentClient(path="database")


def store_chunks(all_chunks, embeddings):
    """
    Store chunks along with their metadata.
    """

    collection = client.get_or_create_collection(
        name="pdf_chatbot"
    )

    ids = [str(uuid.uuid4()) for _ in all_chunks]

    collection.add(
        ids=ids,
        documents=[
            item["text"] for item in all_chunks
        ],
        metadatas=[
            {
                "page": item["page"],
                "chunk": item["chunk"],
                "file": item["file"]
            }
            for item in all_chunks
        ],
        embeddings=embeddings.tolist()
    )


def clear_database():
    """
    Delete the old collection and create a new one.
    """

    try:
        client.delete_collection("pdf_chatbot")
        print("Old collection deleted.")
    except Exception:
        print("No collection found.")

    client.get_or_create_collection(
        name="pdf_chatbot"
    )

    print("New collection created.")