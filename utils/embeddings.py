from sentence_transformers import SentenceTransformer

# Model name as a named constant so other modules (e.g. the sidebar in
# app.py) can display it without hardcoding the string a second time.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Load the embedding model only once
model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def create_embeddings(chunks):
    """
    Convert a list of text chunks into embeddings.
    """

    return model.encode(chunks)