import logging

from tqdm import tqdm

from agentic_rag.retriever.core.types import EmbeddingType
from agentic_rag.retriever.textual.chunker import Chunker
from agentic_rag.retriever.textual.embedder import TextualEmbedder
from agentic_rag.vectordb.handler import QdrantHandler
from scripts.retriever.textual.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_FOLDER,
    DATABASE_NAME,
    DB_HOST,
    DB_PORT,
    EMBEDDER_MODEL_NAME,
    EMBEDDING_DIM,
    TOKENISER_MODEL_NAME,
)
from scripts.utils.dataset import load_vidore_dataset
from scripts.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)
setup_logging()


def main() -> None:
    logger.info("Starting visual database creation...")
    database = QdrantHandler(DB_HOST, DB_PORT)
    database.delete_collection(DATABASE_NAME, False)
    database.create_collection(DATABASE_NAME, EmbeddingType.SINGLE, EMBEDDING_DIM)

    logger.info("Loading vidore dataset...")
    corpus, _, _ = load_vidore_dataset(DATA_FOLDER)
    chunker = Chunker(CHUNK_SIZE, CHUNK_OVERLAP, TOKENISER_MODEL_NAME)
    embedder = TextualEmbedder(EMBEDDER_MODEL_NAME)

    with tqdm(
        corpus,
        desc="Indexing text",
        bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} {postfix}",
        colour="green",
        leave=False,
    ) as corpus_bar:
        for elem in corpus_bar:
            chunks = chunker.chunk_text(elem["markdown"])
            text_embs = embedder.encode_documents(chunks)
            payloads = [
                {
                    "corpus_id": elem["corpus_id"],
                    "doc_id": elem["doc_id"],
                    "page_id": elem["page_number_in_doc"],
                    "chunk_id": i,
                    "content": chunk,
                    "image_path": elem["image_path"],
                }
                for i, chunk in enumerate(chunks)
            ]
            database.add(DATABASE_NAME, text_embs, payloads, False)
    logger.info("Database processing finished !")


if __name__ == "__main__":
    main()
