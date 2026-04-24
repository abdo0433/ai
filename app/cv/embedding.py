# embedding.py

from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# New Chroma Client (no deprecated config)
client = Client(Settings())

# Collection
collection = client.get_or_create_collection(
    name="cv_jd_collection"
)

# Embedder class
class Embedder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, text):
        return self.model.encode([text])[0].tolist()