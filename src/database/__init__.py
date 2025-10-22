"""
Database module for ChromaDB vector storage and retrieval
"""
from .chroma_client import (
    get_chroma_client,
    initialize_collection,
    reset_database,
    get_collection_stats
)
from .chunking import (
    chunk_transcript,
    chunk_by_speaker_turns
)
from .indexing import (
    index_transcript,
    batch_index_transcripts
)

__all__ = [
    # Chroma client functions
    'get_chroma_client',
    'initialize_collection',
    'reset_database',
    'get_collection_stats',

    # Chunking functions
    'chunk_transcript',
    'chunk_by_speaker_turns',

    # Indexing functions
    'index_transcript',
    'batch_index_transcripts'
]
