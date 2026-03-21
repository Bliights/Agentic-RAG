import torch

from agentic_rag.retriever.core.base import BaseRetriever
from agentic_rag.retriever.core.types import VisualResult
from agentic_rag.retriever.visual.embedder import VisualEmbedder
from agentic_rag.vectordb.handler import QdrantHandler


class VisualRetriever(BaseRetriever):
    def __init__(self, collection_name: str, db_handler: QdrantHandler, embedder_name: str) -> None:
        self.collection_name = collection_name
        self.db = db_handler
        self.embedder = VisualEmbedder(embedder_name)

    def search(self, query: str, k: int = 5) -> list[VisualResult]:
        query_emb = self.embedder.encode_query(query)

        matching = self.db.search(
            collection_name=self.collection_name,
            query_vector=self.embedder.to_numpy(query_emb),
            k=k * 5,
        )

        candidates = [
            VisualResult.from_payload(
                r.payload,
                score=r.score,
                vector=torch.tensor(
                    r.vector,
                    dtype=torch.float16,
                    device=self.embedder.device,
                ).unsqueeze(0),
            )
            for r in matching.points
        ]

        for i, _ in enumerate(matching.points):
            new_score = self.embedder.processor.score_multi_vector(query_emb, candidates[i].vector)
            candidates[i].score = new_score.item()

        candidates.sort(key=lambda x: x.score, reverse=True)

        return candidates[:k]
