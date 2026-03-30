import logging

from agentic_rag.pipeline.pipeline import HybridRAGPipeline
from scripts.pipeline.config import (
    DB_HOST,
    DB_PORT,
    LLM_MODEL_NAME,
    SCORER_PATH,
)
from scripts.retriever.textual.config import DATABASE_NAME as TEXTUAL_DATABASE_NAME
from scripts.retriever.textual.config import EMBEDDER_MODEL_NAME as TEXTUAL_EMBEDDER
from scripts.retriever.visual.config import DATABASE_NAME as VISUAL_DATABASE_NAME
from scripts.retriever.visual.config import EMBEDDER_MODEL_NAME as VISUAL_EMBEDDER
from scripts.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)
setup_logging()


def main() -> None:
    """
    Fonction to test the pipeline on a single query
    """
    logger.info("Starting Pipeline evaluation...")
    query = "FDA meeting response times before and during pandemic surge"

    logger.info("Loading pipeline...")
    pipeline = HybridRAGPipeline(
        DB_HOST,
        DB_PORT,
        TEXTUAL_DATABASE_NAME,
        VISUAL_DATABASE_NAME,
        TEXTUAL_EMBEDDER,
        VISUAL_EMBEDDER,
        scorer_model_path=SCORER_PATH,
        llm_model_name=LLM_MODEL_NAME,
    )
    logger.info("Answer generation...")
    result = pipeline.answer(query)
    logger.info("-" * 30 + "PIPELINE RESULT" + "-" * 30)
    logger.info(f"answer : {result.answer}")
    logger.info("docs: ")
    for doc in result.docs:
        logger.info(
            f"\tDoc {doc.doc_id} at page {doc.page_id} : {doc.score:.4f}",
        )


if __name__ == "__main__":
    main()
