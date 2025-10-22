"""
Semantic chunking strategies for transcript segmentation
"""
from typing import List, Dict
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.utils.helpers import seconds_to_timestamp

logger = setup_logger()

def chunk_transcript(transcript: Dict,
                    chunk_duration: float = None,
                    chunk_overlap: float = None,
                    min_chunk_size: int = 50) -> List[Dict]:
    """
    Chunk transcript into semantic segments with speaker context

    Args:
        transcript: Merged transcript dictionary with speaker labels
        chunk_duration: Target duration per chunk in seconds (default: from Config)
        chunk_overlap: Overlap between chunks in seconds (default: from Config)
        min_chunk_size: Minimum characters per chunk

    Returns:
        List of chunk dictionaries with metadata
    """
    chunk_duration = chunk_duration or Config.CHUNK_DURATION
    chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP

    segments = transcript.get("segments", [])
    episode_name = transcript.get("file", "unknown")

    if not segments:
        logger.warning(f"No segments found in transcript for {episode_name}")
        return []

    logger.info(f"Chunking transcript: {episode_name} ({len(segments)} segments)")

    chunks = []
    current_chunk = {
        "text": "",
        "start": None,
        "end": None,
        "speakers": set(),
        "segment_indices": []
    }

    for idx, segment in enumerate(segments):
        segment_start = segment["start"]
        segment_end = segment["end"]
        segment_text = segment["text"]
        segment_speaker = segment.get("speaker", "UNKNOWN")

        # Initialize chunk start time
        if current_chunk["start"] is None:
            current_chunk["start"] = segment_start

        # Check if adding this segment exceeds chunk duration
        chunk_duration_so_far = segment_end - current_chunk["start"]

        if chunk_duration_so_far > chunk_duration and len(current_chunk["text"]) >= min_chunk_size:
            # Save current chunk
            current_chunk["end"] = segment_end
            current_chunk["speakers"] = list(current_chunk["speakers"])
            chunks.append(current_chunk.copy())

            # Start new chunk with overlap
            overlap_start = segment_end - chunk_overlap

            # Find segments that fall within overlap period
            overlap_segments = [
                s for s in segments[max(0, idx-5):idx]
                if s["end"] >= overlap_start
            ]

            current_chunk = {
                "text": " ".join([s["text"] for s in overlap_segments]),
                "start": overlap_start,
                "end": None,
                "speakers": set([s.get("speaker", "UNKNOWN") for s in overlap_segments]),
                "segment_indices": []
            }

        # Add current segment to chunk
        current_chunk["text"] += " " + segment_text
        current_chunk["end"] = segment_end
        current_chunk["speakers"].add(segment_speaker)
        current_chunk["segment_indices"].append(idx)

    # Add final chunk
    if current_chunk["text"].strip() and len(current_chunk["text"]) >= min_chunk_size:
        current_chunk["speakers"] = list(current_chunk["speakers"])
        chunks.append(current_chunk)

    logger.info(f"✓ Created {len(chunks)} chunks from {len(segments)} segments")

    # Format chunks with metadata
    formatted_chunks = []
    for i, chunk in enumerate(chunks):
        formatted_chunk = {
            "chunk_id": i,
            "text": chunk["text"].strip(),
            "start": chunk["start"],
            "end": chunk["end"],
            "timestamp_start": seconds_to_timestamp(chunk["start"]),
            "timestamp_end": seconds_to_timestamp(chunk["end"]),
            "duration": round(chunk["end"] - chunk["start"], 2),
            "speakers": chunk["speakers"],
            "episode": episode_name,
            "num_segments": len(chunk["segment_indices"])
        }
        formatted_chunks.append(formatted_chunk)

    return formatted_chunks

def chunk_by_speaker_turns(transcript: Dict,
                          max_turn_duration: float = 60.0) -> List[Dict]:
    """
    Alternative chunking strategy: chunk by speaker turns

    Args:
        transcript: Merged transcript dictionary
        max_turn_duration: Maximum duration for a single speaker turn

    Returns:
        List of chunk dictionaries
    """
    segments = transcript.get("segments", [])
    episode_name = transcript.get("file", "unknown")

    chunks = []
    current_speaker = None
    current_chunk = {"text": "", "start": None, "end": None, "speaker": None}

    for segment in segments:
        speaker = segment.get("speaker", "UNKNOWN")

        # Check if speaker changed or turn is too long
        if (speaker != current_speaker and current_chunk["text"]) or \
           (current_chunk["start"] and segment["end"] - current_chunk["start"] > max_turn_duration):

            # Save current chunk
            current_chunk["end"] = current_chunk.get("end") or segment["start"]
            chunks.append(current_chunk.copy())

            # Start new chunk
            current_chunk = {
                "text": segment["text"],
                "start": segment["start"],
                "end": segment["end"],
                "speaker": speaker,
                "episode": episode_name
            }
            current_speaker = speaker
        else:
            # Continue current chunk
            if not current_chunk["text"]:
                current_chunk["start"] = segment["start"]
                current_chunk["speaker"] = speaker
                current_speaker = speaker

            current_chunk["text"] += " " + segment["text"]
            current_chunk["end"] = segment["end"]

    # Add final chunk
    if current_chunk["text"].strip():
        chunks.append(current_chunk)

    logger.info(f"✓ Created {len(chunks)} speaker-turn chunks")

    # Format with timestamps
    for i, chunk in enumerate(chunks):
        chunk["chunk_id"] = i
        chunk["timestamp_start"] = seconds_to_timestamp(chunk["start"])
        chunk["timestamp_end"] = seconds_to_timestamp(chunk["end"])
        chunk["duration"] = round(chunk["end"] - chunk["start"], 2)
        chunk["speakers"] = [chunk["speaker"]]

    return chunks
