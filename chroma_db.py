# chroma_db.py: This file will manage the ChromaDB client and operations related to embeddings and storing.

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from config import CHROMA_DB_PATH, DEVICE
from logging_config import logger

# ChromaDB client initialization
def initialize_chroma_client():
    """Initialize ChromaDB client once."""
    global collection
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = chroma_client.get_or_create_collection(name="financial_data")
        logger.info("ChromaDB client initialized successfully.")
        return collection
    except Exception as e:
        logger.error(f"Error initializing ChromaDB client: {e}")
        raise e

# Embedding and storage
def embed_and_store(df, embedding_model, collection):
    """Generate embeddings for the DataFrame and store in ChromaDB."""
    try:
        if df.empty:
            logger.warning("Empty DataFrame passed for embedding.")
            return
        rows_text = df.astype(str).agg(' '.join, axis=1).tolist()
        embeddings = embedding_model.encode(rows_text, convert_to_tensor=True, device=DEVICE).cpu().numpy()
        collection.update(
            ids=[str(i) for i in range(len(embeddings))],
            embeddings=embeddings.tolist(),
            metadatas=[{"row": row} for row in rows_text]
        )
        logger.info("Embeddings stored successfully.")
    except Exception as e:
        logger.error(f"Error storing embeddings: {e}")

# Initialize and load the embedding model
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME).to(DEVICE)
