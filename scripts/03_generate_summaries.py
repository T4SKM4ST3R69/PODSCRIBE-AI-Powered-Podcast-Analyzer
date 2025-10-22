"""
Generate AI summaries for all episodes using Groq
"""
import sys
from pathlib import Path
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag import batch_generate_summaries
from src.utils import Config, setup_logger

logger = setup_logger()


def main():
    """
    Generate summaries for all transcripts
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Generating Episode Summaries")
    logger.info("=" * 60)

    if not Config.GROQ_API_KEY:
        logger.error(" GROQ_API_KEY not configured!")
        logger.error("   Please add your Groq API key to the .env file")
        sys.exit(1)

    logger.info(f"Configuration:")
    logger.info(f"  - LLM Model: {Config.LLM_MODEL}")
    logger.info(f"  - Transcript directory: {Config.TRANSCRIPTS_DIR}")
    logger.info(f"  - Output directory: {Config.SUMMARIES_DIR}")
    logger.info("")

    print(" Note: This will use Groq API credits")
    confirm = input("Continue? (yes/no): ").lower().strip()

    if confirm not in ['yes', 'y']:
        logger.info("Cancelled by user")
        sys.exit(0)

    try:
        start_time = time.time()

        # Batch generate summaries
        stats = batch_generate_summaries(
            transcript_dir=Config.TRANSCRIPTS_DIR,
            output_dir=Config.SUMMARIES_DIR
        )

        elapsed = time.time() - start_time

        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARIZATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total transcripts: {stats['total']}")
        logger.info(f"Successfully generated: {stats['success']}")
        logger.info(f" Failed: {stats['failed']}")
        logger.info(f" Time elapsed: {elapsed / 60:.1f} minutes")

        if stats['failed_files']:
            logger.warning("\nFailed files:")
            for filename in stats['failed_files']:
                logger.warning(f"  - {filename}")

        logger.info(f"\n✓ Summaries saved to: {Config.SUMMARIES_DIR}")

        # List generated summaries
        summaries = list(Config.SUMMARIES_DIR.glob("*.md"))
        if summaries:
            logger.info(f"\nGenerated summaries ({len(summaries)}):")
            for summary in sorted(summaries)[:10]:  # Show first 10
                logger.info(f"  - {summary.name}")
            if len(summaries) > 10:
                logger.info(f"  ... and {len(summaries) - 10} more")

    except Exception as e:
        logger.error(f" Summary generation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
