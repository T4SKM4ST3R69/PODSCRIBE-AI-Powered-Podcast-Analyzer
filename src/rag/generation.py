"""
LLM-based answer generation with Groq - MULTI-SOURCE AWARE
"""
from groq import Groq
from typing import Dict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.rag.retrieval import retrieve_context

logger = setup_logger()


def generate_answer(question: str,
                    context: str = None,
                    model: str = None,
                    temperature: float = None,
                    max_tokens: int = None,
                    top_k: int = None,
                    **retrieval_filters) -> Dict:
    """
    Generate answer using RAG with Groq LLM

    Args:
        question: User question
        context: Pre-retrieved context
        model: Groq model name
        temperature: Sampling temperature
        max_tokens: Maximum response tokens
        top_k: Number of sources to retrieve
        **retrieval_filters: Filters for context retrieval

    Returns:
        Dictionary with answer, context, and metadata
    """
    model = model or Config.LLM_MODEL
    temperature = temperature or Config.LLM_TEMPERATURE
    max_tokens = max_tokens or Config.MAX_TOKENS
    top_k = top_k or Config.TOP_K_RESULTS

    if not Config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not configured. Set in .env file.")

    logger.info(f"Generating answer for: '{question}' (top_k={top_k})")

    # Retrieve context if not provided
    if context is None:
        logger.info("Retrieving context from ALL available sources...")
        context = retrieve_context(
            query=question,
            top_k=top_k,
            **retrieval_filters
        )

    if not context:
        logger.warning("No relevant context found")
        return {
            "question": question,
            "answer": "I couldn't find relevant information in the podcast transcripts. Please upload some podcast files first.",
            "context": "",
            "sources": [],
            "model": model
        }

    # Count unique sources
    source_episodes = set()
    for line in context.split('\n'):
        if line.startswith('[Source') and 'From:' in line:
            episode = line.split('From:')[1].split(']')[0].strip()
            source_episodes.add(episode)

    logger.info(f"Context includes {len(source_episodes)} different podcast(s)")

    # System prompt - Multi-source aware
    system_prompt = """You are an AI assistant specialized in analyzing podcast transcripts from MULTIPLE sources.

CRITICAL INSTRUCTIONS:
1. The context contains segments from MULTIPLE podcast episodes
2. ALWAYS mention which episode you're referencing
3. Include timestamps in format [HH:MM:SS]
4. If information comes from different episodes, make that clear
5. Use speaker names when mentioned
6. Synthesize information across episodes when relevant

Format: "In [Episode Name], the speakers discuss X [00:03:45]. In [Other Episode], they mention Y [00:12:30]."

ALWAYS cite episode name AND timestamp."""

    user_prompt = f"""Based on these podcast transcript segments from MULTIPLE sources, answer the question.

CONTEXT:
{context}

QUESTION: {question}

Provide a detailed answer that:
1. Synthesizes information across all relevant sources
2. Cites specific episodes and timestamps
3. Makes clear when information comes from different podcasts"""

    # Call Groq API
    try:
        client = Groq(api_key=Config.GROQ_API_KEY)

        logger.info(f"Calling Groq API (model: {model})...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        answer = response.choices[0].message.content
        logger.info("✓ Answer generated successfully")

        # Extract source information
        sources = []
        current_source = {}

        for line in context.split('\n'):
            if line.startswith('[Source'):
                if current_source:
                    sources.append(current_source)
                current_source = {}
                if 'From:' in line:
                    episode = line.split('From:')[1].split(']')[0].strip()
                    current_source['episode'] = episode
            elif current_source:
                if line.startswith('Timestamp:'):
                    current_source['timestamp'] = line.split('Timestamp:')[1].strip()
                elif line.startswith('Speakers:'):
                    current_source['speakers'] = line.split('Speakers:')[1].strip()

        if current_source:
            sources.append(current_source)

        result = {
            "question": question,
            "answer": answer,
            "context": context,
            "sources": sources,
            "model": model,
            "temperature": temperature,
            "num_sources": len(sources),
            "num_episodes": len(source_episodes)
        }

        logger.info(f"✓ Answer uses {len(sources)} sources from {len(source_episodes)} episode(s)")
        return result

    except Exception as e:
        logger.error(f"Answer generation failed: {str(e)}")
        return {
            "question": question,
            "answer": f"Error generating answer: {str(e)}",
            "context": context,
            "sources": [],
            "model": model,
            "error": str(e)
        }
