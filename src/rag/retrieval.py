"""
Vector similarity search and context retrieval - MULTI-SOURCE
"""
from typing import List, Dict, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.database.chroma_client import initialize_collection

logger = setup_logger()


def search_transcripts(query: str,
                       top_k: int = None,
                       episode_filter: Optional[str] = None,
                       speaker_filter: Optional[str] = None,
                       collection_name: str = None) -> List[Dict]:
    """
    Search transcripts using vector similarity

    Args:
        query: Search query
        top_k: Number of results to return (default: from Config)
        episode_filter: Filter by episode name (optional)
        speaker_filter: Filter by speaker name (optional)
        collection_name: ChromaDB collection name (default: from Config)

    Returns:
        List of search results with metadata
    """
    top_k = top_k or Config.TOP_K_RESULTS
    collection = initialize_collection(collection_name)

    logger.info(f"Searching for: '{query}' (top_k={top_k})")

    # Build metadata filter
    where_filter = {}
    if episode_filter:
        where_filter["episode"] = {"$eq": episode_filter}
    if speaker_filter:
        where_filter["speakers"] = {"$contains": speaker_filter}

    # Perform vector search
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where_filter if where_filter else None,
        include=["documents", "metadatas", "distances"]
    )

    # Format results
    formatted_results = []

    for i in range(len(results['ids'][0])):
        result = {
            "id": results['ids'][0][i],
            "text": results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            "similarity_score": 1 - results['distances'][0][i],
            "episode": results['metadatas'][0][i].get('episode', 'unknown'),
            "timestamp_start": results['metadatas'][0][i].get('timestamp_start', '00:00:00'),
            "timestamp_end": results['metadatas'][0][i].get('timestamp_end', '00:00:00'),
            "speakers": results['metadatas'][0][i].get('speakers', '').split(',')
        }
        formatted_results.append(result)

    logger.info(f"✓ Found {len(formatted_results)} results")

    if formatted_results:
        scores = [r['similarity_score'] for r in formatted_results]
        logger.info(f"Similarity range: {min(scores):.3f} - {max(scores):.3f}")

    return formatted_results


def retrieve_context(query: str,
                     top_k: int = None,
                     episode_filter: Optional[str] = None,
                     **filters) -> str:
    """
    Retrieve and format context from ALL sources

    Args:
        query: Search query
        top_k: Number of results to retrieve
        episode_filter: Optional episode filter
        **filters: Additional metadata filters

    Returns:
        Formatted context string for LLM
    """
    top_k = top_k or Config.TOP_K_RESULTS

    results = search_transcripts(
        query=query,
        top_k=top_k,
        episode_filter=episode_filter,
        **filters
    )

    if not results:
        logger.warning(f"No results found for query: '{query}'")
        return ""

    # Group by episode
    episodes_found = set([r['episode'] for r in results])
    logger.info(f"Retrieved {len(results)} results from {len(episodes_found)} episode(s)")

    # Format context with source attribution
    context_parts = []
    for i, result in enumerate(results, 1):
        speakers_str = ", ".join(result['speakers'])
        score = result.get('similarity_score', 0)

        context_part = (
            f"[Source {i} - From: {result['episode']}]\n"
            f"Timestamp: {result['timestamp_start']} - {result['timestamp_end']}\n"
            f"Speakers: {speakers_str}\n"
            f"Relevance: {score:.2%}\n"
            f"Content: {result['text']}\n"
        )
        context_parts.append(context_part)

    context = "\n\n".join(context_parts)

    logger.info(f"✓ Context built from {len(episodes_found)} different episode(s)")

    return context
