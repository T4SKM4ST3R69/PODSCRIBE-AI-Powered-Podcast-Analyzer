"""
RAG (Retrieval-Augmented Generation) module for question answering and summarization
"""
from .retrieval import (
    search_transcripts,
    retrieve_context
)
from .generation import (
    generate_answer
)
from .summarization import (
    generate_episode_summary,
    batch_generate_summaries
)

__all__ = [
    # Retrieval functions
    'search_transcripts',
    'retrieve_context',

    # Generation functions
    'generate_answer',

    # Summarization functions
    'generate_episode_summary',
    'batch_generate_summaries'
]
