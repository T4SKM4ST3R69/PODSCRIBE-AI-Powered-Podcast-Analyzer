"""
ChromaDB indexing with ACCURATE timestamp chunking
"""
from typing import Dict, List
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.utils.helpers import seconds_to_timestamp, load_json
from src.database.chroma_client import initialize_collection

logger = setup_logger()


def create_chunks_with_accurate_timestamps(transcript: Dict,
                                           chunk_duration: int = None,
                                           chunk_overlap: int = None) -> List[Dict]:
    """
    Create chunks with ACCURATE timestamp preservation

    Args:
        transcript: Merged transcript dictionary
        chunk_duration: Duration of each chunk in seconds
        chunk_overlap: Overlap between chunks in seconds

    Returns:
        List of chunk dictionaries with accurate timestamps
    """
    chunk_duration = chunk_duration or Config.CHUNK_DURATION
    chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP

    segments = transcript["segments"]
    chunks = []

    current_chunk = {
        "text": "",
        "start": None,
        "end": None,
        "speakers": set()
    }

    for segment in segments:
        segment_start = segment["start"]
        segment_end = segment["end"]
        segment_text = segment["text"]
        segment_speaker = segment["speaker"]

        # Initialize first chunk
        if current_chunk["start"] is None:
            current_chunk["start"] = segment_start

        # Check if adding this segment exceeds chunk duration
        chunk_duration_so_far = segment_end - current_chunk["start"]

        if chunk_duration_so_far > chunk_duration and current_chunk["text"]:
            # Save current chunk with ACCURATE end timestamp
            current_chunk["end"] = segment_start  # End at start of new segment
            current_chunk["speakers"] = list(current_chunk["speakers"])
            chunks.append(current_chunk.copy())

            # Start new chunk with overlap
            overlap_start = max(0, segment_start - chunk_overlap)
            current_chunk = {
                "text": segment_text,
                "start": overlap_start,  # Start with overlap
                "end": None,
                "speakers": {segment_speaker}
            }
        else:
            # Add to current chunk
            current_chunk["text"] += " " + segment_text if current_chunk["text"] else segment_text
            current_chunk["speakers"].add(segment_speaker)
            current_chunk["end"] = segment_end  # Update end timestamp

    # Add final chunk
    if current_chunk["text"]:
        current_chunk["speakers"] = list(current_chunk["speakers"])
        chunks.append(current_chunk)

    logger.info(f"Created {len(chunks)} chunks with accurate timestamps")

    # Format chunks with precise timestamp strings
    formatted_chunks = []
    for i, chunk in enumerate(chunks):
        formatted_chunks.append({
            "id": f"{transcript['file']}_{i}",
            "text": chunk["text"].strip(),
            "start": round(chunk["start"], 3),
            "end": round(chunk["end"], 3),
            "timestamp_start": seconds_to_timestamp(chunk["start"]),
            "timestamp_end": seconds_to_timestamp(chunk["end"]),
            "speakers": chunk["speakers"],
            "episode": transcript["file"]
        })

    return formatted_chunks


def index_transcript(transcript: Dict,
                    collection_name: str = None) -> int:
    """
    Index transcript into ChromaDB with accurate timestamps

    Args:
        transcript: Merged transcript dictionary
        collection_name: ChromaDB collection name

    Returns:
        Number of chunks indexed
    """
    collection = initialize_collection(collection_name)
    episode_name = transcript["file"]

    logger.info(f"Indexing transcript: {episode_name}")

    # Create chunks with accurate timestamps
    chunks = create_chunks_with_accurate_timestamps(transcript)

    if not chunks:
        logger.warning("No chunks created from transcript")
        return 0

    # Prepare data for ChromaDB
    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = []

    for chunk in chunks:
        metadata = {
            "episode": chunk["episode"],
            "timestamp_start": chunk["timestamp_start"],
            "timestamp_end": chunk["timestamp_end"],
            "start_seconds": chunk["start"],
            "end_seconds": chunk["end"],
            "speakers": ",".join(chunk["speakers"]),
            "num_speakers": len(chunk["speakers"])
        }
        metadatas.append(metadata)

    # Add to collection
    try:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        logger.info(f"✓ Indexed {len(chunks)} chunks with accurate timestamps")
        return len(chunks)

    except Exception as e:
        logger.error(f"Failed to index transcript: {str(e)}")
        raise


def batch_index_transcripts(transcript_dir: Path = None,
                           collection_name: str = None) -> Dict:
    """
    Batch index all transcripts with accurate timestamps

    Args:
        transcript_dir: Directory containing transcript JSON files
        collection_name: ChromaDB collection name

    Returns:
        Dictionary with indexing statistics
    """
    transcript_dir = Path(transcript_dir) if transcript_dir else Config.TRANSCRIPTS_DIR
    collection = initialize_collection(collection_name)

    transcript_files = list(transcript_dir.glob("*.json"))

    if not transcript_files:
        logger.warning(f"No transcript files found in {transcript_dir}")
        return {"total": 0, "success": 0, "failed": 0, "total_chunks": 0}

    logger.info(f"Batch indexing {len(transcript_files)} transcripts")

    success_count = 0
    total_chunks = 0
    failed_files = []

    for i, transcript_file in enumerate(transcript_files, 1):
        logger.info(f"[{i}/{len(transcript_files)}] Processing {transcript_file.name}")

        try:
            transcript = load_json(transcript_file)
            chunks_indexed = index_transcript(transcript, collection_name)
            total_chunks += chunks_indexed
            success_count += 1

        except Exception as e:
            logger.error(f"Failed to index {transcript_file.name}: {str(e)}")
            failed_files.append(transcript_file.name)
            continue

    stats = {
        "total": len(transcript_files),
        "success": success_count,
        "failed": len(failed_files),
        "total_chunks": total_chunks,
        "failed_files": failed_files
    }

    logger.info(f"✓ Batch indexing complete: {stats}")
    return stats
