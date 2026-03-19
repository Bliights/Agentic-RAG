import logging

from tqdm import tqdm

from agentic_rag.retriever.core.types import EmbeddingType
from agentic_rag.retriever.visual.model import VisualModel
from agentic_rag.vectordb.handler import QdrantHandler
from scripts.retriever.visual.config import (
    DATA_FOLDER,
    DATABASE_NAME,
    DB_HOST,
    DB_PORT,
    EMBEDDING_DIM,
    MODEL_NAME,
)
from scripts.utils.dataset import load_vidore_dataset
from scripts.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)
setup_logging()


def main() -> None:
    logger.info("Starting visual database creation...")
    database = QdrantHandler(DB_HOST, DB_PORT)
    logger.info("Loading vidore dataset...")
    corpus, _, _ = load_vidore_dataset(DATA_FOLDER)
    model = VisualModel(MODEL_NAME)

    logger.info("Start of database creation...")
    database.delete_collection(DATABASE_NAME, False)
    database.create_collection(DATABASE_NAME, EmbeddingType.MULTI, EMBEDDING_DIM)
    with tqdm(
        corpus,
        desc="Indexing images",
        bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} {postfix}",
        colour="green",
        leave=False,
    ) as corpus_bar:
        for elem in corpus_bar:
            img_emb = model.encode_image(elem["image"])
            payload = {
                "corpus_id": elem["corpus_id"],
                "doc_id": elem["doc_id"],
                "page_id": elem["page_number_in_doc"],
                "image_path": elem["image_path"],
            }
            database.add(DATABASE_NAME, [model.to_numpy(img_emb)], [payload], False)
    logger.info("Database processing finished !")


if __name__ == "__main__":
    main()
