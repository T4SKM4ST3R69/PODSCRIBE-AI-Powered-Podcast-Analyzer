"""
Build ChromaDB vector database from transcripts
"""
import sys
from pathlib import Path
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.database import batch_index_transcripts, get_collection_stats
from src.utils import Config, setup_logger

logger = setup_logger()


def main():
    """
    Index all transcripts into ChromaDB
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Building Vector Database")
    logger.info("=" * 60)

    # Ask if user wants to reset the database
    print("\n  Do you want to reset the existing database?")
    print("   This will DELETE all existing data!")
    reset = input("   Reset database? (yes/no): ").lower().strip()

    reset_db = reset in ['yes', 'y']

    if reset_db:
        logger.warning("  Resetting database...")
    else:
        logger.info("Appending to existing database...")

    logger.info(f"\nConfiguration:")
    logger.info(f"  - Collection: {Config.COLLECTION_NAME}")
    logger.info(f"  - Embedding model: {Config.EMBEDDING_MODEL}")
    logger.info(f"  - Chunk duration: {Config.CHUNK_DURATION}s")
    logger.info(f"  - Chunk overlap: {Config.CHUNK_OVERLAP}s")
    logger.info(f"  - Device: {Config.DEVICE}")
    logger.info("")

    try:
        start_time = time.time()

        # Batch index all transcripts
        stats = batch_index_transcripts(
            transcript_dir=Config.TRANSCRIPTS_DIR,
            collection_name=Config.COLLECTION_NAME,
            reset_collection=reset_db
        )

        elapsed = time.time() - start_time

        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("INDEXING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total transcript files: {stats['total_files']}")
        logger.info(f" Successfully indexed: {stats['indexed_files']}")
        logger.info(f" Failed: {len(stats['failed_files'])}")
        logger.info(f" Total chunks created: {stats['total_chunks']}")
        logger.info(f"  Time elapsed: {elapsed:.1f}s")

        if stats['failed_files']:
            logger.warning("\nFailed files:")
            for filename in stats['failed_files']:
                logger.warning(f"  - {filename}")

        # Get collection statistics
        logger.info("\n" + "=" * 60)
        logger.info("DATABASE STATISTICS")
        logger.info("=" * 60)

        collection_stats = get_collection_stats()
        logger.info(f"Collection: {collection_stats['collection_name']}")
        logger.info(f"Total chunks: {collection_stats['total_chunks']}")
        logger.info(f"Embedding model: {collection_stats['embedding_model']}")

        if collection_stats['sample_episodes']:
            logger.info(f"\nSample episodes (showing first 5):")
            for episode in collection_stats['sample_episodes'][:5]:
                logger.info(f"  - {episode}")

        if collection_stats['sample_speakers']:
            logger.info(f"\nSpeakers found:")
            for speaker in collection_stats['sample_speakers']:
                logger.info(f"  - {speaker}")

        logger.info(f"\n Vector database ready at: {Config.CHROMA_DB_DIR}")

    except Exception as e:
        logger.error(f" Database building failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
