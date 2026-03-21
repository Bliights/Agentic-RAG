import numpy as np

from agentic_rag.retriever.core.base import BaseRetriever
from agentic_rag.retriever.core.types import TextualResult
from agentic_rag.retriever.textual.embedder import TextualEmbedder
from agentic_rag.vectordb.handler import QdrantHandler


class TextualRetriever(BaseRetriever):
    def __init__(self, collection_name: str, db_handler: QdrantHandler, embedder_name: str) -> None:
        self.collection_name = collection_name
        self.db = db_handler
        self.embedder = TextualEmbedder(embedder_name)

    def search(self, query: str, k: int = 5) -> list[TextualResult]:
        query_vector = self.embedder.encode_query(query)

        matching = self.db.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            k=k,
        )

        return [
            TextualResult.from_payload(
                r.payload,
                score=r.score,
                vector=np.array(r.vector, dtype=np.float32),
            )
            for r in matching.points
        ]
