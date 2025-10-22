"""
Episode summarization using Groq LLM
"""
from groq import Groq  # Changed from openai import OpenAI
from typing import Dict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.utils.helpers import load_json, save_json

logger = setup_logger()


def generate_episode_summary(transcript: Dict,
                             model: str = None,
                             max_tokens: int = 8000,  # Increased for Groq
                             output_path: Path = None) -> str:
    """
    Generate comprehensive episode summary using Groq LLM

    Args:
        transcript: Merged transcript dictionary
        model: Groq model name (default: from Config)
        max_tokens: Maximum summary tokens
        output_path: Optional path to save summary markdown

    Returns:
        Summary text in markdown format
    """
    model = model or Config.LLM_MODEL

    if not Config.GROQ_API_KEY:  # Changed from OPENAI_API_KEY
        raise ValueError("GROQ_API_KEY not configured. Set in .env file.")

    episode_name = transcript.get("file", "unknown")
    logger.info(f"Generating summary for: {episode_name}")

    # Prepare transcript text
    segments = transcript.get("segments", [])

    # Sample segments if too long (every Nth segment)
    max_segments = 150  # Increased from 100 since Groq has larger context
    if len(segments) > max_segments:
        step = len(segments) // max_segments
        sampled_segments = segments[::step]
        logger.info(f"Sampling {len(sampled_segments)} segments from {len(segments)} total")
    else:
        sampled_segments = segments

    transcript_text = "\n\n".join([
        f"[{seg.get('speaker', 'UNKNOWN')}]: {seg['text']}"
        for seg in sampled_segments
    ])

    # Construct prompt
    system_prompt = """You are an expert podcast summarizer. Create a comprehensive, engaging summary of the podcast episode.

Your summary should include:
1. **Title**: Create a catchy episode title
2. **Overview**: 2-3 sentence overview of the main topic
3. **Key Points**: Bullet points of main discussion points
4. **Notable Quotes**: 2-3 interesting quotes with speaker attribution
5. **Takeaways**: Key insights or action items

Format in markdown with proper headers and structure."""

    user_prompt = f"""Please summarize this podcast episode transcript:

{transcript_text}

Episode file: {episode_name}
Total segments: {len(segments)}
Speakers: {', '.join(transcript.get('speakers', []))}

Create a comprehensive summary following the format specified."""

    # Call Groq API
    try:
        client = Groq(api_key=Config.GROQ_API_KEY)  # Changed from OpenAI

        logger.info(f"Calling Groq API for summarization (model: {model})...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=max_tokens
        )

        summary = response.choices[0].message.content

        logger.info("✓ Summary generated successfully")

        # Save to file if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            logger.info(f"Saved summary to {output_path}")

        return summary

    except Exception as e:
        logger.error(f"Summary generation failed: {str(e)}")
        raise


def batch_generate_summaries(transcript_dir: Path = None,
                             output_dir: Path = None) -> Dict:
    """
    Batch generate summaries for all transcripts

    Args:
        transcript_dir: Directory containing transcript JSON files
        output_dir: Directory for output summaries

    Returns:
        Dictionary with generation statistics
    """
    transcript_dir = Path(transcript_dir) if transcript_dir else Config.TRANSCRIPTS_DIR
    output_dir = Path(output_dir) if output_dir else Config.SUMMARIES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript_files = list(transcript_dir.glob("*.json"))

    if not transcript_files:
        logger.warning(f"No transcript files found in {transcript_dir}")
        return {"total": 0, "success": 0, "failed": 0}

    logger.info(f"Generating summaries for {len(transcript_files)} transcripts")

    success_count = 0
    failed_files = []

    for i, transcript_file in enumerate(transcript_files, 1):
        logger.info(f"[{i}/{len(transcript_files)}] Processing {transcript_file.name}")

        try:
            transcript = load_json(transcript_file)
            output_path = output_dir / f"{transcript_file.stem}_summary.md"

            # Skip if summary already exists
            if output_path.exists():
                logger.info(f"Summary already exists: {output_path.name}")
                success_count += 1
                continue

            generate_episode_summary(transcript, output_path=output_path)
            success_count += 1

        except Exception as e:
            logger.error(f"Failed to generate summary for {transcript_file.name}: {str(e)}")
            failed_files.append(transcript_file.name)
            continue

    stats = {
        "total": len(transcript_files),
        "success": success_count,
        "failed": len(failed_files),
        "failed_files": failed_files
    }

    logger.info(f"✓ Batch summarization complete: {stats}")
    return stats
