import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from agentic_rag.retriever.core.base import BaseEmbedder


class TextualEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = SentenceTransformer(
            model_name,
            device=self.device,
        )

    def encode_query(self, query: str) -> np.ndarray:
        query = f"query: {query}"
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding[0]

    def encode_documents(self, texts: list[str]) -> list[np.ndarray]:
        texts = [f"passage: {t}" for t in texts]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return list(embeddings)
