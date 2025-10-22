
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger()

# Singleton client instance
_chroma_client = None

def get_chroma_client():
    """
    Get or create ChromaDB persistent client (Singleton pattern)

    Returns:
        ChromaDB PersistentClient instance
    """
    global _chroma_client

    if _chroma_client is None:
        logger.info(f"Initializing ChromaDB client at {Config.CHROMA_DB_DIR}")

        _chroma_client = chromadb.PersistentClient(
            path=str(Config.CHROMA_DB_DIR),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        logger.info("✓ ChromaDB client initialized")

    return _chroma_client

def initialize_collection(collection_name: str = None,
                         embedding_model: str = None,
                         reset: bool = False):
    """
    Initialize or get ChromaDB collection with sentence-transformers embeddings

    Args:
        collection_name: Name of the collection (default: from Config)
        embedding_model: Sentence-transformers model name (default: from Config)
        reset: If True, delete existing collection and create new one

    Returns:
        ChromaDB Collection instance
    """
    collection_name = collection_name or Config.COLLECTION_NAME
    embedding_model = embedding_model or Config.EMBEDDING_MODEL

    client = get_chroma_client()

    # Delete existing collection if reset requested
    if reset:
        try:
            client.delete_collection(name=collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass  # Collection doesn't exist

    # Create embedding function using sentence-transformers
    logger.info(f"Loading embedding model: {embedding_model}")
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model,
        device=Config.DEVICE
    )

    # Get or create collection
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"description": "Podcast transcripts with speaker diarization"}
        )

        count = collection.count()
        logger.info(f"✓ Collection '{collection_name}' ready ({count} documents)")

        return collection

    except Exception as e:
        logger.error(f"Failed to initialize collection: {str(e)}")
        raise

def reset_database():
    """
    Reset the entire ChromaDB database (delete all collections)
    """
    client = get_chroma_client()

    logger.warning("Resetting ChromaDB database...")
    collections = client.list_collections()

    for collection in collections:
        client.delete_collection(name=collection.name)
        logger.info(f"Deleted collection: {collection.name}")

    logger.info("✓ Database reset complete")

def get_collection_stats(collection_name: str = None) -> dict:

    collection_name = collection_name or Config.COLLECTION_NAME
    collection = initialize_collection(collection_name)

    count = collection.count()

    # Sample a few documents to get metadata info
    sample = collection.get(limit=10)

    episodes = set()
    speakers = set()

    if sample['metadatas']:
        for metadata in sample['metadatas']:
            if 'episode' in metadata:
                episodes.add(metadata['episode'])
            if 'speaker' in metadata:
                speakers.add(metadata['speaker'])

    stats = {
        "collection_name": collection_name,
        "total_chunks": count,
        "sample_episodes": list(episodes),
        "sample_speakers": list(speakers),
        "embedding_model": Config.EMBEDDING_MODEL
    }

    logger.info(f"Collection stats: {stats}")
    return stats
